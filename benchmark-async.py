import asyncio
import aiohttp
import os
from dotenv import load_dotenv
import argparse
import logging
import time
import json
from datetime import datetime, timezone

# Load environment variables from .env file
load_dotenv()

# test with GBB cluster
# API_ENDPOINT_ENV = os.environ["GPT_4o_SWC_20240913_ENDPOINT"]
# DEPLOYMENT = os.environ["GPT_4o_SWC_20240913_PTU_DEPLOYMENT"]
# # DEPLOYMENT = os.environ["GPT_4o_SWC_20240913_GLOBAL_PAYGO_DEPLOYMENT"]
# API_KEY = os.environ["GPT_4o_SWC_20240913_KEY"]

# test with my JP east endpoint
API_ENDPOINT_ENV = os.environ["PAYGO_4o_GLOBAL_JP_ENDPOINT"]
DEPLOYMENT = os.environ["PAYGO_4o_GLOBAL_JP_DEPLOYMENT"]
API_KEY = os.environ["PAYGO_4o_GLOBAL_JP_KEY"]

# Parse command-line arguments
parser = argparse.ArgumentParser(description="AOAI Request Benchmark Script")
# parser.add_argument('--prompt', type=str, required=True, help='Prompt to send to the model')
parser.add_argument('--tokens', type=int, required=True, help='Number of tokens to generate')
parser.add_argument('--requests', type=int, default=100, help='Total number of requests to send')
parser.add_argument('--rpm', type=int, default=20, help='Requests per minute')
parser.add_argument('--openai', type=str.lower, choices=['true', 'false'], default='false', help='Set to true to use OpenAI API')
parser.add_argument(
        '--stream',
        type=str.lower,
        choices=['true', 'false'],
        default='false',
        help='Enable (true) or disable (false) streaming requests'
    )

args = parser.parse_args()
tokens = args.tokens
use_stream = args.stream == 'true'
use_openai = args.openai == 'true'

# Azure OpenAI API settings
API_VERSION = "2024-06-01"  # Ensure this is the correct API version
API_ENDPOINT = f"{API_ENDPOINT_ENV}/openai/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}&generated_tokens={tokens}"
HEADERS = {
    "Content-Type": "application/json",
    "api-key": API_KEY
}

prompt = f"Generate {tokens} tokens in Japanese about banks, of which {tokens//4} about Japan bank history, {tokens//4} about different businesses that a bank in Japan runs, {tokens//4}  about operations and processes, {tokens//4}  about new technologies. Don’t stop until the requested token count is reached."

PAYLOAD = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant trained to generate exactly the number of tokens requested. Always generate content with deep detail and explanation to ensure the required token count is reached. Don't output anything else."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": tokens,
        "temperature": 0.7,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

# OpenAI API settings

if use_openai:
    OPENAI_API_ENDPOINT = f"{os.getenv("OPENAI_API_ENDPOINT")}?tokens={tokens}"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API Key is required when using OpenAI.")
    API_ENDPOINT = OPENAI_API_ENDPOINT 
    HEADERS = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

# Configure logging
logging.basicConfig(
    filename=f"aoai_benchmark_openai_{tokens}tokens{"_strean" if use_stream else ""}.log" if use_openai else f"aoai_benchmark_{tokens}tokens{"_stream" if use_stream else ""}.log",
    filemode='w',  # Overwrite the log file each time the script runs
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

## Without streaming
async def make_request(session, i, data):
    request_time = datetime.now(timezone.utc)
    request_time_str = request_time.isoformat()

    try:
        async with session.post(API_ENDPOINT, headers=HEADERS, json=data) as response:
            if response.status == 200:
                result = await response.json()
                
                response_time = datetime.now(timezone.utc)
                response_time_str = datetime.now(timezone.utc).isoformat()
                
                duration = (response_time - request_time).total_seconds()
                
                content = result["choices"][0]["message"]["content"]
                completion_tokens = result["usage"]["completion_tokens"]
                
                logging.info(
                    f"Request {i} | Endpoint: {API_ENDPOINT} | Status: {response.status} | Request Time: {request_time_str} | Response Time: {response_time_str} |  Duration: {duration:.3f}s | Completion tokens: {completion_tokens} | Response: {content}" 
                )
                
                print(f'Request {i} | Endpoint: {API_ENDPOINT} | Status: {response.status} | Request Time: {request_time_str} | Response Time: {response_time_str} | Duration: {duration:.3f}s | Completion tokens: {completion_tokens}')
                return duration
            else:
                error_text = await response.text()
                
                response_time = datetime.now(timezone.utc)
                response_time_str = datetime.now(timezone.utc).isoformat()
                
                duration = (response_time - request_time).total_seconds()
                
                logging.error(
                    f"Request {i} | Endpoint: {API_ENDPOINT} | Status: {response.status} | Request Time: {request_time_str} | Response Time: {response_time_str} | Duration: {duration:.3f}s | Error: {error_text}"
                )
                print(
                    f"Request {i} | Endpoint: {API_ENDPOINT} | Status: {response.status} | Request Time: {request_time_str} | Response Time: {response_time_str} | Duration: {duration:.3f}s | Error: {error_text}"
                )
                return None
                
    except Exception as e:
        error_time = datetime.now(timezone.utc)
        error_time_str = error_time.isoformat()
        duration = (error_time - request_time).total_seconds() 
        
        logging.exception(
            f"Request {i} | Endpoint: {API_ENDPOINT} | Status: Exception | Request Time: {request_time_str} | Exception Time: {error_time_str} | Duration: {duration:.3f} | Exception: {e}"
        )
        print(
            f"Request {i} | Endpoint: {API_ENDPOINT} | Status: Exception | Request Time: {request_time_str} | Exception Time: {error_time_str} | Duration: {duration:.3f} | Exception: {e}"
        )
        return None
        


## With streaming
async def make_request_stream(session, i, data):
    request_time = datetime.now(timezone.utc)
    request_time_str = request_time.isoformat()
    first_token_time = None
    done_time = None
    
    try:
        async with session.post(API_ENDPOINT, headers=HEADERS, json=data) as response:
            if response.status == 200:
                while True:
                    line = await response.content.readline()
                    if not line:
                        break
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line.startswith('data: '):
                        data_str = decoded_line[6:]
                        if data_str == '[DONE]':
                            done_time = datetime.now(timezone.utc)
                            break
                        else:
                            try:
                                data_json = json.loads(data_str)
                                # Two lines will be received before first token, one with empty choices and one with empty choices["delta"]["content"]:
                                ## {"choices":[],"created":0,"id":"","model":"","object":"","prompt_filter_results":[{"prompt_index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"},"jailbreak":{"filtered":false,"detected":false},"self_harm":{"filtered":false,"severity":"safe"},"sexual":{"filtered":false,"severity":"safe"},"violence":{"filtered":false,"severity":"safe"}}}]}
                                ## {"choices":[{"content_filter_results":{},"delta":{"content":"","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1726396815,"id":"chatcmpl-A7gxr5My7tpoYHvihqTysGy150dHI","model":"gpt-4","object":"chat.completion.chunk","system_fingerprint":null}
                                choices = data_json.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content.strip() != '':
                                        if first_token_time is None:
                                            first_token_time = datetime.now(timezone.utc)
                            except json.JSONDecodeError as e:
                                logging.error(f"Reauest {i} | Endpoint: {API_ENDPOINT} | Error decoding JSON: {e}")
                                print(f"Request {i} | Endpoint: {API_ENDPOINT} | Error decoding JSON: {e}")
                                # terminate current streaming request 
                                return (None, None)
                # After streaming is done, calculate the total time
                if first_token_time:
                    first_token_time_str = first_token_time.isoformat()
                    time_to_first_token = (first_token_time - request_time).total_seconds()
                else:
                    time_to_first_token = None
                    first_token_time_str = None
                if done_time:
                    done_time_str = done_time.isoformat()
                    total_time = (done_time - request_time).total_seconds()
                else:
                    total_time = None
                    done_time_str = None
                logging.info(
                    f"Request {i} | Endpoint: {API_ENDPOINT} | Status: {response.status} | Request Time: {request_time_str} | First Token Time: {first_token_time_str} | Done Time: {done_time_str} | Time to First Token: {time_to_first_token:.3f}s | Total Time: {total_time:.3f}s"
                )
                print(
                    f"Request {i} | Endpoint: {API_ENDPOINT} | Status: {response.status} | Request Time: {request_time_str} | First Token Time: {first_token_time_str} | Done Time: {done_time_str} | Time to First Token: {time_to_first_token:.3f}s | Total Time: {total_time:.3f}s"
                )
                return (time_to_first_token, total_time)
            # response status is not 200
            else:
                error_text = await response.text()
                error_time = datetime.now(timezone.utc)
                error_time_str = error_time.isoformat()
                logging.error(
                    f"Request {i} | Endpoint: {API_ENDPOINT} | Status: {response.status} | Request Time: {request_time_str} | Error Time: {error_time_str} | Error: {error_text}"
                )
                print(
                    f"Request {i} | Endpoint: {API_ENDPOINT} | Status: {response.status} | Request Time: {request_time_str} | Error Time: {error_time_str} | Error: {error_text}"
                )
                return (None, None)
    except Exception as e:
        error_time = datetime.now(timezone.utc)
        error_time_str = error_time.isoformat()
        logging.exception(
            f"Request {i} | Endpoint: {API_ENDPOINT} | Status: Exception | Request Time: {request_time_str} | Exception Time: {error_time_str} | Exception: {e}"
        )
        print(
            f"Request {i} | Endpoint: {API_ENDPOINT} | Status: Exception | Request Time: {request_time_str} | Exception Time: {error_time_str} | Exception: {e}"
        )
        return (None, None)                                        
                            
async def schedule_request(session, request_number, data, delay):
    await asyncio.sleep(delay)
    if use_stream:
        # if stream is true, use make_request_stream()
        # and the returned the duration will be a tuple of (time_to_first_token, total_time)
        duration = await make_request_stream(session, request_number, data)
    else:
        duration = await make_request(session, request_number, data)
    return duration

async def scheduler(session, total_requests, data, delay_between_calls):
    tasks = []
    for i in range(total_requests):
        request_start_time = i * delay_between_calls
        task = asyncio.create_task(schedule_request(session, i+1, data, request_start_time))
        tasks.append(task)
    durations = await asyncio.gather(*tasks) # execute all tasks concurrently
    return durations

async def main():
    delay_between_calls = 60 / args.rpm  # seconds
    total_requests = args.requests
    
    data = PAYLOAD
    if use_openai:
        data["model"] = "gpt-4o"
    if use_stream:
        data["stream"] = True

    logging.info(f"Starting benchmark: Tokens: {tokens} | Requests: {total_requests} | RPM: {args.rpm}")
    async with aiohttp.ClientSession() as session:
        durations = await scheduler(session, total_requests, data, delay_between_calls)
        if use_stream:
            # durations is a lit of tuples: (time_to_first_token, total_time)
            time_to_first_token_list = [d[0] for d in durations if d[0] is not None]
            total_time_list = [d[1] for d in durations if d[1] is not None]
            
            avg_time_to_first_token = sum(time_to_first_token_list) / len(time_to_first_token_list) if time_to_first_token_list else None
            avg_time_to_first_token_str = f"{avg_time_to_first_token:.3f}s" if avg_time_to_first_token else "N/A"
            
            avg_total_time = sum(total_time_list) / len(total_time_list) if total_time_list else None
            avg_total_time_str = f"{avg_total_time:.3f}s" if avg_total_time else "N/A"
            
            print(f"Successful requests made: {len(total_time_list)} | Average Time to First Token: {avg_time_to_first_token_str}s | Average Total Time: {avg_total_time_str}s | RPM: {args.rpm}")
            logging.info(f"Successful requests made: {len(total_time_list)} | Average Time to First Token: {avg_time_to_first_token_str}s | Average Total Time: {avg_total_time_str}s | RPM: {args.rpm}")
        else:
            # durations is a list of durations for each request
            success_durations = [d for d in durations if d is not None]
            if success_durations:
                average_duration = sum(success_durations) / len(success_durations)
                print(f"Successful requests made: {len(success_durations)} | Average duration: {average_duration:.3f}s | RPM: {args.rpm}")
                logging.info(f"Successful requests made: {len(success_durations)} | Average duration: {average_duration:.3f}s | RPM: {args.rpm}")
            else:
                print("No successful requests were made.")
                logging.info("No successful requests were made.")

if __name__ == '__main__':
    asyncio.run(main())
