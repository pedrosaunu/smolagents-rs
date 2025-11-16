# 🤖 smolagents-rs

This is a rust implementation of HF [smolagents](https://github.com/huggingface/smolagents) library. It provides a powerful autonomous agent framework written in Rust that solves complex tasks using tools and LLM models. 

---

## ✨ Features

- 🧠 **Function-Calling Agent Architecture**: Implements the ReAct framework for advanced reasoning and action.
- 🔍 **Built-in Tools**:
  - Google Search
  - DuckDuckGo Search
  - Website Visit & Scraping
  - Wikipedia Search
  - Tree-sitter Code Parser (Rust, Python, JavaScript, Bash)
- 🤝 **OpenAI Integration**: Works seamlessly with GPT models.
- 🎯 **Task Execution**: Enables autonomous completion of complex tasks.
- 🔄 **State Management**: Maintains persistent state across steps.
- 📊 **Beautiful Logging**: Offers colored terminal output for easy debugging.

---

![demo](https://res.cloudinary.com/dltwftrgc/image/upload/v1737485304/smolagents-small_fmaikq.gif)

## ✅ Feature Checklist

### Models

- [x] OpenAI Models (e.g., GPT-4o, GPT-4o-mini)
- [x] Azure OpenAI support
- [x] Ollama Integration
- [x] Hugging Face API support
 - [x] Open-source model integration via Candle
- [x] Light LLM integration

### Agents

- [x] Tool-Calling Agent
- [x] CodeAgent
- [x] Planning Agent

The code agent is still in development, so there might be python code that is not yet supported and may cause errors. Try using the tool-calling agent for now.

### Tools

- [x] Google Search Tool
- [x] DuckDuckGo Tool
- [x] Website Visit & Scraping Tool
- [x] RAG Tool
- [x] Wikipedia Search Tool
- [x] Tree-sitter Code Parser Tool (multi-language)
- More tools to come...

### Other

 - [x] Sandbox environment
- [x] Streaming output
- [x] Improve logging
- [x] Parallel execution

## Known Issues

- `cargo fmt` and `cargo clippy` require additional components that may not be installable in restricted environments.
- The browser example depends on the `wasm32-unknown-unknown` target and `wasm-pack` which may fail to install without network access.

---

## TODO Checklist

- [x] Implement ReAct step logic for `MultiStepAgent`
- [x] Handle truncated observations and decide if they should produce a final answer
- [x] Support additional tool call types
- [x] Support other log variants when streaming
 - [x] Propagate parsing errors instead of returning `None`
 - [x] Handle unexpected step log types
- [x] Return the final answer when available

## Remaining Tasks

- [x] Decide if truncated observations should yield a final answer (see TODO in `src/agents.rs`)
- [x] Remove unreachable pattern branches returning `Ok(None)` in `src/agents.rs`
- [x] Replace many `unwrap()` calls in `GoogleSearchTool` with proper error handling
 - [x] Add JavaScript and Bash tests for the `TreeSitterTool`
- [x] Implement streaming support for Candle, HuggingFace and LightLLM models


## 🚀 Quick Start

The agent can run inside a temporary sandbox directory by passing `--sandbox` or setting the `SANDBOX_DIR` environment variable.

### Using Docker

```bash
# Pull the image
docker pull akshayballal95/smolagents-rs:latest

# Run with your OpenAI API key
docker run -e OPENAI_API_KEY=your-key-here smolagents-rs -t "What is the latest news about Rust programming?"
```

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/smolagents-rs.git
cd smolagents-rs

# Build the project
cargo build --release --features cli-deps

# Run the agent
OPENAI_API_KEY=your-key-here ./target/release/smolagents-rs -t "Your task here"
```

---

## 🛠️ Usage

```bash
smolagents-rs [OPTIONS] -t TASK

Options:
  -t, --task <TASK>          The task to execute
  -a, --agent-type <TYPE>    Agent type [default: function-calling]
  -l, --tools <TOOLS>        Comma-separated list of tools [default: duckduckgo,visit-website]
  -m, --model <TYPE>         Model type [default: open-ai]
  -k, --api-key <KEY>        API key for OpenAI, Hugging Face, or LightLLM models
  --model-id <ID>            Model ID (e.g., "gpt-4" for OpenAI or "qwen2.5" for Ollama) [default: gpt-4o-mini]
  # For Azure OpenAI use --model azure-open-ai and pass your deployment ID as --model-id
  -u, --ollama-url <URL>     Ollama server URL [default: http://localhost:11434]
  -s, --stream               Enable streaming output
  --sandbox                  Run in an isolated sandbox directory
  -h, --help                 Print help
```

---

## 🌟 Examples

```bash
# Simple search task
smolagents-rs -t "What are the main features of Rust 1.75?"

# Research with multiple tools
smolagents-rs -t "Compare Rust and Go performance" -l duckduckgo,google-search,visit-website

# Stream output for real-time updates
smolagents-rs -t "Analyze the latest crypto trends" -s

# Run multiple tasks in parallel
cargo run --example parallel --features cli,code-agent
# Compile to WebAssembly for browser usage
wasm-pack build examples/browser --release
serve examples/browser
# Then open `index.html` to try the interactive multi-language AST demo
```

---

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required).
- `AZURE_OPENAI_API_KEY`: API key for Azure OpenAI service (optional).
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL (optional).
- `AZURE_OPENAI_DEPLOYMENT_ID`: Deployment ID for your chat model (optional).
- `AZURE_OPENAI_API_VERSION`: API version for Azure OpenAI (optional).
- `SERPAPI_API_KEY`: Google Search API key (optional).
- `HF_API_KEY`: Hugging Face API key (optional).
- `CANDLE_MODEL_PATH`: Path to a local Candle model directory.
- `LIGHTLLM_API_KEY`: API key for LightLLM server (optional).
- `SANDBOX_DIR`: Directory for creating the sandbox when `--sandbox` is used.

## 📡 Using smolagents with FUSE

`smolagents-rs` can be pointed at the FUSE gateway/runtime that was described in the original RFP. Read [docs/fuse.md](docs/fuse.md) for the in-depth architecture notes, then configure the CLI with the following knobs:

### Required environment variables

- `FUSE_GATEWAY_URL`: Base URL for the gateway (REST + SSE endpoints).
- `FUSE_API_KEY`: Bearer token used for manuals, runs, and uploads.
- `FUSE_MANUAL_ID`: Manual to fetch and verify before tool hydration.
- `FUSE_MANUAL_PUBKEY`: Public key (Base64) that validates the signed manual.
- `FUSE_FLOW_ID` / `FUSE_DEPLOYMENT_ID`: Orchestration objects that the agent should target when submitting runs.

### Optional environment variables

- `FUSE_PEER_MODE`: Set to `1` to enable peer networking.
- `FUSE_PEER_PRIVATE_KEY` / `FUSE_PEER_ATTESTATION`: Required when `FUSE_PEER_MODE` is enabled so the gateway will forward Libp2p traffic.
- `FUSE_CHUNK_SIZE`: Override the Base64 chunk size (defaults to 256 KiB) when uploading large tool artifacts.
- `FUSE_CACHE_DIR`: Override the default cache path (`~/.config/smolagents/fuse`) for manuals and run transcripts.

### CLI flags

- `--fuse-flow <ID>` and `--fuse-deployment <ID>` select the flow/deployment pair. When omitted, the runtime falls back to the environment variables above.
- `--fuse-peer-mode` toggles peer routing; combine it with the peer environment variables so that the agent can register its keypair.
- `--emit-fuse-gui-links` prints handy URLs for the official GUI (e.g., `https://console.fuse.local/runs/<id>`), mirroring what the web console would display.
- `smolagents-rs fuse manual download --manual-id <id> --pubkey $FUSE_MANUAL_PUBKEY` is a helper command (available when the CLI features are enabled) that lets you prefetch and verify a manual before running a flow.

### Runtime limitations

- Every request/response exchanged with FUSE must be Base64-encoded. The CLI automatically does this, but tool authors must keep payload sizes manageable or rely on chunking.
- The sandboxed runtime blocks arbitrary outbound networking. Only gateway, manual, and peer endpoints are reachable, so third-party APIs must be exposed as managed FUSE tools.
- Runs fail fast when the manual signature cannot be verified. Double-check `FUSE_MANUAL_PUBKEY` whenever onboarding a new tenant.
- Flow retries will reuse the cached run metadata plus `parent_run_id` so the gateway and CLI timelines stay aligned. Keep that cache directory persisted if you want full post-mortems.

### Example invocation

```bash
FUSE_GATEWAY_URL=https://api.fuse.local \
FUSE_API_KEY=sk-live... \
FUSE_MANUAL_ID=customer-manual \
FUSE_MANUAL_PUBKEY=$(cat manual.pub) \
FUSE_FLOW_ID=customer-flow \
FUSE_DEPLOYMENT_ID=customer-deployment \
smolagents-rs --features cli-deps --fuse-flow customer-flow --fuse-deployment customer-deployment -t "Review the signed manual and summarize available tools"
```

Refer to [docs/fuse.md](docs/fuse.md) for diagrams and a deeper explanation of how manuals, flows, deployments, runs, and peer mode map onto `smolagents-rs` abstractions.

---

## 🏗️ Architecture

The project follows a modular architecture with the following components:

1. **Agent System**: Implements the ReAct framework.

2. **Tool System**: An extensible tool framework for seamless integration of new tools.

3. **Model Integration**: Robust OpenAI API integration for powerful LLM capabilities.

---

## 🚀 Why port to Rust?
Rust provides critical advantages that make it the ideal choice for smolagents-rs:

1. ⚡ **High Performance**:<br>
Zero-cost abstractions and no garbage collector overhead enable smolagents-rs to handle complex agent tasks with near-native performance. This is crucial for running multiple agents and processing large amounts of data efficiently.

2. 🛡️ **Memory Safety & Security**:<br>
Rust's compile-time guarantees prevent memory-related vulnerabilities and data races - essential for an agent system that handles sensitive API keys and interacts with external resources. The ownership model ensures thread-safe concurrent operations without runtime overhead.

3. 🔄 **Powerful Concurrency**:<br>
Fearless concurrency through the ownership system enable smolagents-rs to efficiently manage multiple agents and tools in parallel, maximizing resource utilization.

4. 💻 **Universal Deployment**:<br>
Compile once, run anywhere - from high-performance servers to WebAssembly in browsers. This allows smolagents-rs to run natively on any platform or be embedded in web applications with near-native performance.

Apart from this, its essential to push new technologies around agentic systems to the Rust ecoystem and this library aims to do so. 

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add some amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.


---

## ⭐ Show Your Support

Give a ⭐️ if this project helps you or inspires your work!

