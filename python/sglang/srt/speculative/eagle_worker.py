import logging
import time
from typing import List, Optional, Union

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_utils import (
    EagleDraftInput,
    EagleVerifyInput,
    assign_draft_cache_locs,
    fast_topk,
    select_top_k_tokens,
)
from sglang.srt.speculative.eagle_mab import MABGroupManager, MetricsEntry

logger = logging.getLogger(__name__)


class EAGLEWorker(TpModelWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Do not capture cuda graph in `super().__init__()`
        # We will capture it later
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        super().__init__(
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            server_args=server_args,
            nccl_port=nccl_port,
            dp_rank=dp_rank,
            is_draft_worker=True,
        )
        self.target_worker = target_worker
        self.finish_extend_len = []

        # Parse arguments
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.server_args = server_args

        # Share the embedding and lm_head
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        self.model_runner.model.set_embed_and_head(embed, head)
        self.model_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph

        # Create multi-step attn backends and cuda graph runners
        from sglang.srt.layers.attention.flashinfer_backend import (
            FlashInferMultiStepDraftBackend,
        )

        self.default_mab_strategy = self.get_current_mab_strategy()
        self.speculative_eagle_mab = self.server_args.speculative_eagle_mab.split(',')
        if self.speculative_eagle_mab:
            self.mab_algorithm = self.speculative_eagle_mab[0]  # Default to Epsilon-Greedy
            self.mab_strategies = self.speculative_eagle_mab[1:] + [self.default_mab_strategy]
            self.mab_strategies = sorted(list(set(self.mab_strategies)))
        else:
            self.mab_algorithm = 'EB' # epsilon_greedy
            self.mab_strategies = [self.default_mab_strategy]
            self.speculative_eagle_mab = f'{self.mab_algorithm},{self.default_mab_strategy}'

        # Initialize MAB settings
        self.mab_window_size = getattr(server_args, 'speculative_mab_window_size', 300)
        self.groups = list(range(1,32)) + list(range(32, 128, 8)) + list(range(128, 257, 32))

        # Initialize MAB manager
        self.mab_manager = MABGroupManager(
            groups=self.mab_groups,
            strategies=self.mab_strategies,
            algorithm=self.mab_algorithm,
            window_size=self.mab_window_size
        )

        # Initialize timing events for performance monitoring
        self.mab_last_pull = {
            "usage_count": 0,
            "mab_strategy": None,
            "batch_size": None,
            "accept_length_avg": None,
            "is_decode": False,
            "events": [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        }

        self.max_topk = self.topk
        self.draft_attn_backends = dict()
        self.cuda_graph_runners = dict()
        self.target_worker.model_runner.attn_backends = dict()
        self.target_worker.model_runner.cuda_graph_runners = dict()

        # Initialize cuda graph runners for each MAB strategy
        self.target_worker.model_runner.attn_backends[self.default_mab_strategy] = self.target_worker.model_runner.attn_backend
        self.target_worker.model_runner.cuda_graph_runners[self.default_mab_strategy] = self.target_worker.model_runner.cuda_graph_runner
        self.last_mab_strategy = self.default_mab_strategy

        for mab_strategy in self.mab_strategies:
            self.set_mab_strategy(mab_strategy)

            # Draft worker
            self.max_topk = max(self.max_topk, self.topk)
            self.draft_attn_backend = FlashInferMultiStepDraftBackend(
                self.model_runner,
                self.topk,
                self.speculative_num_steps,
            )
            self.model_runner.draft_attn_backend = self.draft_attn_backend
            self.init_cuda_graphs()

            self.draft_attn_backends[mab_strategy] = self.draft_attn_backend
            self.cuda_graph_runners[mab_strategy] = self.cuda_graph_runner

            # Target worker
            if mab_strategy == self.default_mab_strategy:
                pass
            else:
                # repeat the same process for the target worker
                self.target_worker.model_runner.init_attention_backend()
                self.target_worker.model_runner.init_cuda_graphs()

                self.target_worker.model_runner.attn_backends[mab_strategy] = self.target_worker.model_runner.attn_backend
                self.target_worker.model_runner.cuda_graph_runners[mab_strategy] = self.target_worker.model_runner.cuda_graph_runner

        # Set default MAB strategy
        self.set_mab_strategy(self.default_mab_strategy)

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None

        if self.server_args.disable_cuda_graph:
            return

        tic = time.time()
        logger.info("Capture cuda graph begin. This can take up to several minutes.")
        self.cuda_graph_runner = EAGLEDraftCudaGraphRunner(self)
        logger.info(f"Capture cuda graph end. Time elapsed: {time.time() - tic:.2f} s")

    def get_current_mab_strategy(self) -> str:
        """Get MAB strategy string from current speculative decoding settings."""
        mab_strategy = f"{self.server_args.speculative_num_steps}_{self.server_args.speculative_eagle_topk}_{self.server_args.speculative_num_draft_tokens}"
        return mab_strategy

    def set_mab_strategy(self, mab_strategy: str):
        """Apply MAB strategy by updating speculative decoding settings.
        
        Args:
            mab_strategy: String in format 'steps_topk_tokens'
        """
        if len(self.mab_strategies) == 1:
            return
        if mab_strategy == self.last_mab_strategy:
            return

        self.last_mab_strategy = mab_strategy

        steps, topk, draft_tokens = map(int, mab_strategy.split("_"))

        # Update the setting for the draft worker
        self.speculative_num_steps = self.server_args.speculative_num_steps = steps
        self.topk = self.server_args.speculative_eagle_topk = topk
        self.server_args.speculative_num_draft_tokens = draft_tokens

        # Update the setting for the draft worker's cuda graph runner
        self.cuda_graph_runner = self.cuda_graph_runners.get(mab_strategy, None)
        self.draft_attn_backend = self.draft_attn_backends.get(mab_strategy, None)
        self.model_runner.draft_attn_backend = self.draft_attn_backend

        # Target worker
        if self.target_worker.model_runner.cuda_graph_runner:
            self.target_worker.model_runner.server_args.speculative_num_steps = self.server_args.speculative_num_steps
            self.target_worker.model_runner.server_args.speculative_eagle_topk = self.server_args.speculative_eagle_topk
            self.target_worker.model_runner.server_args.speculative_num_draft_tokens = self.server_args.speculative_num_draft_tokens

            self.target_worker.model_runner.attn_backend = self.target_worker.model_runner.attn_backends.get(mab_strategy, None)
            self.target_worker.model_runner.cuda_graph_runner = self.target_worker.model_runner.cuda_graph_runners.get(mab_strategy, None)


    def select_mab_strategy(self, batch: ScheduleBatch) -> str:
        """Select and apply MAB strategy for the given batch.
        
        Args:
            batch: Batch of requests to process
            
        Returns:
            Selected MAB strategy string
        """
        bs = len(batch.reqs)
        mab_strategy = self.mab_manager.select_strategy(bs)
        self.set_mab_strategy(mab_strategy)
        return mab_strategy

    def record_mab_strategy_metrics(self, mab_last_pull):
        torch.cuda.synchronize()
        events = mab_last_pull["events"]

        elapsed_seconds = [0] * (len(events) - 1)
        for i in range(len(events)-1):
            elapsed_seconds[i] = events[i].elapsed_time(events[i+1]) / 1000.0

        mab_strategy = mab_last_pull["mab_strategy"]
        accept_length_avg = mab_last_pull["accept_length_avg"]
        bs = mab_last_pull["batch_size"]

        if len(self.mab_strategies) > 1:
            start = time.perf_counter()
            stable_accept_length = self.mab_manager.get_stable_accept_length(mab_strategy)
            mab_time = time.perf_counter() - start
        else:
            stable_accept_length = accept_length_avg
            mab_time = 0.0
        
        # Calculate metrics
        total_time = sum(elapsed_seconds)
        metrics_entry = MetricsEntry(
            reward=stable_accept_length * bs / total_time,
            goodput=accept_length_avg * bs / total_time,
            accept_length=accept_length_avg,
            total_time=total_time,
            draft_time=elapsed_seconds[0],
            verify_time=elapsed_seconds[1],
            draft_extend_time=elapsed_seconds[2],
            other_time=elapsed_seconds[3],
        )
        metrics_entry.additional_metrics["mab_time"] = mab_time
        
        # Update metrics in MAB manager
        self.mab_manager.add_single_step_metrics(bs, mab_strategy, metrics_entry)

    def forward_batch_speculative_generation(self, batch: ScheduleBatch):
        events = self.mab_last_pull["events"]
        if self.mab_last_pull["is_decode"]:
            # Speculative Decoding does support overlap-schedule yet. This is to measure the 
            # overhead due to non-overlapped scheduler time since last decoding step
            events[4].record()
            self.record_mab_strategy_metrics(self.mab_last_pull)

        self.mab_last_pull["is_decode"] = False
        if batch.forward_mode.is_decode():
            self.mab_last_pull["is_decode"] = True

            self.mab_last_pull["mab_strategy"] = self.select_mab_strategy(batch)
            batch.spec_info.topk_p = batch.spec_info.topk_p[:, :self.topk]
            batch.spec_info.topk_index = batch.spec_info.topk_index[:, :self.topk]

            # Draft
            events[0].record()
            spec_info: EagleVerifyInput = self.draft(batch)

            # Verify
            events[1].record()
            (
                next_draft_input,
                logits_output,
                verified_id,
                self.finish_extend_len,
                accept_length_cpu,
                model_worker_batch,
            ) = self.verify(batch, spec_info)
            batch.spec_info = next_draft_input

            # if it is None, means all requsets are finished
            events[2].record()
            if batch.spec_info.verified_id is not None:
                self.forward_draft_extend_after_decode(batch)

            events[3].record()
            self.mab_last_pull["accept_length_avg"] = sum(accept_length_cpu) / len(batch.reqs) + 1
            self.mab_last_pull["batch_size"] = len(batch.reqs)

            return (
                logits_output,
                verified_id,
                model_worker_batch,
                sum(accept_length_cpu),
            )

        else:
            # Forward with the target model and get hidden states.
            # We need the full hidden states to prefill the KV cache of the draft model.
            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            logits_output, next_token_ids = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            # Forward with the draft model.
            batch.spec_info = EagleDraftInput(
                hidden_states=logits_output.hidden_states,
                verified_id=next_token_ids,
            )
            self.forward_draft_extend(batch)
            return logits_output, next_token_ids, model_worker_batch, 0

    def draft(self, batch: ScheduleBatch):
        self._set_mem_pool(batch, self.model_runner)

        # Parse args
        num_seqs = batch.batch_size()
        spec_info = batch.spec_info

        # Allocate cache locations
        out_cache_loc = batch.alloc_token_slots(
            num_seqs * self.topk * self.speculative_num_steps
        )
        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
        )

        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)

        # Get forward batch
        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        can_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(
            forward_batch
        )

        if can_cuda_graph:
            score_list, token_list, parents_list = self.cuda_graph_runner.replay(
                forward_batch
            )
        else:
            # Initialize attention backend
            self.draft_attn_backend.init_forward_metadata(forward_batch)

            # Run forward steps
            score_list, token_list, parents_list = self.draft_forward(forward_batch)

        # SGlang has a bug here: The model might output nan in score_list after a certain 
        # speculative_num_steps for some reqs in the batch. 
        # Ever observed one case where 3 out of 145 reqs have nan in their score_list 
        # when speculative_num_steps = 4. 
        for i in range(self.speculative_num_steps):
            isnan_tokens = torch.isnan(score_list[i])
            if torch.any(isnan_tokens):
                # Identify the reqs that include nan values
                isnan_reqs = isnan_tokens.view(num_seqs, -1).max(dim=1).values

                # Set to equal probability among child tokens, so that the chain rule is not violate
                min_score = 1 if i == 0 else score_list[i-1][isnan_reqs].min()
                for j in range(i, self.speculative_num_steps):
                    score_list[j][isnan_reqs] = min_score / self.topk
                    min_score = min_score / self.topk

        ret = EagleVerifyInput.create(
            spec_info.verified_id,
            score_list,
            token_list,
            parents_list,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.server_args.speculative_num_draft_tokens,
        )

        # Free cache locations
        batch.token_to_kv_pool.free(out_cache_loc)
        self._set_mem_pool(batch, self.target_worker.model_runner)
        return ret

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # This is copied from a later sglang PR
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[
                forward_batch.batch_size
                * self.topk
                * i : forward_batch.batch_size
                * self.topk
                * (i + 1)
            ]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output = self.model_runner.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            hidden_states = logits_output.hidden_states

        return score_list, token_list, parents_list

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        spec_info.prepare_for_verify(batch)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = spec_info
        model_worker_batch = batch.get_model_worker_batch()
        logits_output, _ = self.target_worker.forward_batch_generation(
            model_worker_batch, skip_sample=True
        )
        spec_info.hidden_states = logits_output.hidden_states
        res = spec_info.verify(batch, logits_output)
        batch.forward_mode = ForwardMode.DECODE
        return res + (model_worker_batch,)

    def forward_draft_extend(self, batch: ScheduleBatch):
        self._set_mem_pool(batch, self.model_runner)
        batch.spec_info.prepare_for_extend(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        self.capture_for_decode(logits_output, forward_batch)
        self._set_mem_pool(batch, self.target_worker.model_runner)

    def _set_mem_pool(self, batch: ScheduleBatch, runner: ModelRunner):
        batch.token_to_kv_pool = runner.token_to_kv_pool
        batch.req_to_token_pool = runner.req_to_token_pool

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        seq_lens_backup = batch.seq_lens

        self._set_mem_pool(batch, self.model_runner)
        batch.forward_mode = ForwardMode.DRAFT_EXTEND
        batch.spec_info.prepare_extend_after_decode(batch, self.speculative_num_steps)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        self.capture_for_decode(logits_output, forward_batch)
        self._set_mem_pool(batch, self.target_worker.model_runner)

        # Restore backup.
        # This is because `seq_lens` can be modified in `prepare_extend_after_decode`
        batch.forward_mode = ForwardMode.DECODE
        batch.seq_lens = seq_lens_backup

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, forward_batch: ForwardBatch
    ):
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        spec_info = forward_batch.spec_info
        # Change from self.topk to self.max_topk, so that it can support MAB with different topk values
        spec_info.topk_p, spec_info.topk_index = fast_topk(probs, self.max_topk, dim=-1)
        spec_info.hidden_states = logits_output.hidden_states

    # Don't support prefix share now.
    def finish_request(self, reqs: Union[Req, List[Req]]):
        if not isinstance(reqs, List):
            reqs = [reqs]
        for req in reqs:
            if req.rid not in self.finish_extend_len:
                continue
            req_len = (
                len(req.origin_input_ids)
                + len(req.output_ids)
                - self.finish_extend_len[req.rid]
                - 1
            )
            kv_indices = self.model_runner.req_to_token_pool.req_to_token[
                req.req_pool_idx
            ][:req_len]
            self.model_runner.token_to_kv_pool.free(kv_indices)
            self.model_runner.req_to_token_pool.free(req.req_pool_idx)
