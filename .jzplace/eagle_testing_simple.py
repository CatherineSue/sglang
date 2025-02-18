import time
import requests
import statistics  # for computing mean and standard deviation

# This is to test the impact of 1) applying chat template ([INST] and [/INST]) in case of Llama3.1-8B models; 2) different types of data (speculative decoding is best at fact checking /coding data)
# The draft model is finetuned with chat template. When it is applied, the draft and target model will predict more consistent results
prompts = {
    "joke": "Tell me a long joke in 1000 words",
    "code": "Give me a simple FastAPI server. Show the python code.",
    "fact": "Tell me about Seatlle in 1000 words",
    "creative": "Write a creative story about a cat in 1000 words",
}

prompts_templated = {
    f"{key}-template": f"[INST] {prompt} [/INST]" for key, prompt in prompts.items()
}

prompts = {**prompts_templated, **prompts}

ports = {
    "w_speculative": 30022,
    "wo_speculative": 30023,
}
avg_speed = {}
std_speed = {}

# Higher temperature will lead to lower consistency between draft and target model
for temperature in [0, 2]:
    print(f"\n\nTemperature: {temperature}")
    for prompt_type, prompt in prompts.items():
        for port_type, port in ports.items():
            speeds = []
            for i in range(3):
                # Log iteration start
                # print(f"Starting iteration {i+1}")

                tic = time.time()
                
                response = requests.post(
                    f"http://localhost:{port}/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "temperature": temperature,
                            "max_new_tokens": 256,
                            "ignore_eos": True,
                        },
                    },
                )
                
                latency = time.time() - tic
                ret = response.json()
                
                # Print the complete response text
                # print(ret["text"])
                
                # Calculate and print the speed for this iteration
                speed = ret["meta_info"]["completion_tokens"] / latency
                # print(f"Iteration {i+1} speed: {speed:.2f} token/s")
                
                speeds.append(speed)

            # Compute and print average and standard deviation of speed across iterations
            avg_speed[f"{prompt_type},{port_type},{temperature}"] = statistics.mean(speeds)
            std_speed[f"{prompt_type},{port_type},{temperature}"] = statistics.stdev(speeds)

            print(f"{prompt_type} with {port_type}: {statistics.mean(speeds):.2f} +/- {statistics.stdev(speeds):.2f} token/s")
