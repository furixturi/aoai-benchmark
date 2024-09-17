## About
A (quick and dirty) script to benchmark GPT-4o chat completion and compare the <b>time to first token</b> and <b>total request time</b> between:
- Azure OpenAI PTU
- Azure OpenAI Global Standard PayGo
- OpenAI API

for generating 500, 1000, 2000, 3000, 4000 tokens, respectively.

## Supported Parameters and Options
When running the script you can specify the following parameters
- `--tokens`: int. Required. How many tokens to generate in each request
- `--requests`: int. Default to 100. How many requests to execute.
- `--rpm`: int. Default to 20. How many requests to make per minute. Adjust this according to your TPM rate limit to avoid 429 errors.
- `--stream`: `true` or `false`. Default to false. Whether to use streaming so that you also get stats for TTFT (time to first token). 
- `--openai`: `true` or `false`. Default to false. Whether to use OpenAI endpoint.

## How to use
- (Optional) Create a Python virtual environment and activate it.
- Run `pip install -r requirement.txt`
- Add a `.env` file where you specify your environment variables for your API endpoint and API key (Azure OpenAI or OpenAI). For Azure OpenAI, also specify an environment variable for the deployment name.
- Adjust the `benchmark-async.py` script to load your environment variables to the `API_ENDPOINT_ENV`, `DEPLOYMENT`, `API_KEY`, `OPENAI_API_ENDPOINT`, `OPENAI_API_KEY`.
- Run the script with the parameters you need. For example
```bash
python benchmark-async.py --tokens 500 --requests 10 --rpm 50 --stream true --openai true
```
