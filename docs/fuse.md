# FUSE Integration Guide

This document summarizes the expected FUSE platform architecture and how `smolagents-rs` consumes the capabilities that were outlined in the RFP. It is meant to help maintainers reason about the interoperability contract before the runtime and SDK layers are finalized.

## 1. Platform Overview

| Area | Key Ideas from the RFP |
| --- | --- |
| Gateway & Manual APIs | The gateway exposes REST/WS endpoints for initiating runs, polling orchestration state, and downloading a *signed manual* that lists the tools, schemas, and deployment constraints for a specific customer. Manuals are versioned (`manual_id`, `manual_version`) and the gateway signs each payload; SDKs are expected to verify signatures locally before trusting capabilities. |
| Runtime Constraints | Payloads must be delivered as Base64 strings, even when chaining sub-requests, because the runtime sandboxes every invocation and only accepts binary-safe, deterministic blobs. Tool execution happens inside an ephemeral sandbox (Firecracker-style VM or container) with read-only root, short-lived `/tmp`, and outbound traffic restricted to the gateway. |
| Orchestration Objects | **Flows** describe reusable DAGs of tool calls, **Deployments** bind a flow to concrete runtime resources and access policies, and **Runs** track an agent invocation of a deployment. The gateway enforces that SDKs target a deployment ID and operate on the resulting run IDs when streaming logs or uploading chained payloads. |
| Peer Mode | Besides managed deployments, FUSE supports a peer-to-peer cooperative mode where trusted agents exchange signed tool responses over Libp2p. A *peer attestation token* plus the agent's public key is required before the gateway will forward peer traffic. |
| SDK Expectations | SDKs must fetch the manual before submitting any work, hydrate strongly typed tool schemas, and respect per-tool concurrency quotas. Every outbound payload must carry the manual digest, deployment ID, run ID, and SDK version metadata. |
| GUI Requirements | The reference GUI consumes the same APIs: it lists manuals, allows triggering flows, inspects run timelines, and displays Base64 artifacts via download links. SDKs should expose equivalent metadata to keep UX parity (e.g., URL to download the Base64 transcript, decoded tool summaries, peer mode status). |

## 2. How `smolagents-rs` consumes FUSE capabilities

### 2.1 Manual acquisition and verification

1. `smolagents-rs` calls `GET {FUSE_GATEWAY_URL}/manuals/{manual_id}` using `FUSE_API_KEY`.
2. The response body contains `manual_bytes_base64` plus `signature_base64`.
3. The CLI/SDK decodes the bytes, verifies the detached signature with the configured `FUSE_MANUAL_PUBKEY`, and caches the resulting manual in `~/.config/smolagents/fuse/manuals/{manual_id}-{version}.json`.
4. The verified manual is converted into internal [`ToolInfo`](../src/tools/mod.rs) structs:
   - `manual.tools[].schema` → `ToolInfo.schema` JSON.
   - `manual.tools[].runtime` → `ToolInfo.runtime_constraints` metadata to enforce Base64-only arguments.
   - `manual.tools[].manual_url` → stored as `documentation_url` for tracing.

### 2.2 Submitting Base64 tool chains

- Agents build observations/actions exactly as they would for local tools, but the executor wraps the payload in the FUSE envelope:
  ```json
  {
    "deployment_id": "FUSE_DEPLOYMENT_ID",
    "run_id": "auto-generated UUID",
    "manual_digest": "sha256...",
    "payload_base64": "...",
    "parent_run_id": "optional when chaining"
  }
  ```
- The `--fuse-flow` CLI flag (or `FUSE_FLOW_ID`) identifies which flow DAG the agent is targeting. The CLI automatically uploads chained Base64 responses back to `{FUSE_GATEWAY_URL}/runs/{run_id}/steps` using the `FUSE_API_KEY` bearer token.
- Runtime logs are streamed via Server-Sent Events from `{FUSE_GATEWAY_URL}/runs/{run_id}/events`, allowing the agent loop to surface gateway sandbox output to end users.

### 2.3 Peer mode handshake

- Passing `--fuse-peer-mode` (or setting `FUSE_PEER_MODE=1`) instructs the runtime to register the agent's peer identity with the gateway using:
  - `FUSE_PEER_PRIVATE_KEY`: local Libp2p key used to sign peer messages.
  - `FUSE_PEER_ATTESTATION`: short-lived token issued by the gateway admin plane.
- Once acknowledged, tool calls targeting peers serialize the action as Base64, sign it, and publish through the gateway broker. The receiving peer validates the signature, decodes the request, runs it inside its sandbox, and responds through the same channel. `smolagents-rs` exposes these events as synthetic tools named `fuse-peer::<peer_id>` so planners can reason about them like any other capability.

### 2.4 GUI compatibility hooks

- Each run exposes a JSON summary at `~/.config/smolagents/fuse/runs/{run_id}.json` with:
  - Manual metadata (ID, version, digest, download URL).
  - Tool invocations plus decoded arguments/results.
  - Links to the Base64 artifacts stored in `runs/{run_id}/artifacts/*`.
- CLI flag `--emit-fuse-gui-links` prints GUI-friendly URLs (e.g., `https://console.fuse.local/runs/{run_id}`) so that operators can jump into the official dashboard.

### 2.5 Failure handling and limitations

- Missing manuals or signature validation failures abort the agent run before any tool call is attempted.
- Because the runtime only accepts Base64 payloads, binary responses must be chunked (`FUSE_CHUNK_SIZE` defaults to 256 KiB) and reassembled client-side.
- The sandbox cannot reach arbitrary hosts; tools that need the public internet must be modeled as managed FUSE tools or proxied through peers.

## 3. Required configuration

| Variable / Flag | Purpose |
| --- | --- |
| `FUSE_GATEWAY_URL` | Base URL for the FUSE gateway (e.g., `https://api.fuse.local`). |
| `FUSE_API_KEY` | Bearer token for gateway/manual APIs. |
| `FUSE_MANUAL_ID` | Manual to fetch during startup. |
| `FUSE_MANUAL_PUBKEY` | Base64-encoded public key used to verify the manual signature. |
| `FUSE_FLOW_ID` / `--fuse-flow` | Flow (DAG) identifier the agent should target. |
| `FUSE_DEPLOYMENT_ID` / `--fuse-deployment` | Concrete deployment that hosts the flow. |
| `FUSE_PEER_MODE` / `--fuse-peer-mode` | Enables peer-to-peer routing. |
| `FUSE_PEER_PRIVATE_KEY` | Signing key for peer mode. |
| `FUSE_PEER_ATTESTATION` | Gateway-issued attestation token authorizing peer traffic. |
| `FUSE_CHUNK_SIZE` | Optional override for Base64 chunk size. |
| `--emit-fuse-gui-links` | Adds GUI URLs to CLI output for quick inspection. |

Refer back to this document whenever a new capability lands so the assumptions stay aligned with the gateway contract.
