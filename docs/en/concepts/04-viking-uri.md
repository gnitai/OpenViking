# Viking URI

Viking URI is the unified resource identifier for all content in OpenViking.

## Format

```
wfs://{scope}/{path}
```

- **scheme**: Always `viking`
- **scope**: Top-level namespace (`resources`, `user`, `agent`, `session`; `temp` and `queue` are internal)
- **path**: Resource path within the scope

## Scopes

| Scope | Description | Lifecycle | Visibility |
|-------|-------------|-----------|------------|
| **resources** | Independent resources | Long-term | Global |
| **user** | User-level data | Long-term | Global |
| **agent** | Agent-level data | Long-term | Global |
| **session** | Session-level data | Session lifetime | Current session |
| **queue** | Processing queue | Temporary | Internal |
| **temp** | Temporary files | During parsing | Internal |

Public API and CLI filesystem/content operations accept only the public scopes:
`resources`, `user`, `agent`, and `session` (plus the root URI `wfs://`).
`temp` and `queue` are internal implementation scopes and cannot be addressed
directly through public API URI parameters.

## Initial Directory Structure

Moving away from traditional flat database thinking, all context is organized as a filesystem. Agents no longer just find data through vector search, but can locate and browse data through deterministic paths and standard filesystem commands. Each context or directory is assigned a unique URI identifier string in the format wfs://{scope}/{path}, allowing the system to precisely locate and access resources stored in different locations.

```
wfs://
├── session/{session_id}/
│   ├── .abstract.md          # L0: One-line session summary
│   ├── .overview.md          # L1: Session overview
│   ├── .meta.json            # Session metadata
│   ├── messages.json         # Structured message storage
│   ├── checkpoints/          # Version snapshots
│   ├── summaries/            # Compression summary history
│   └── .relations.json       # Relations table
│
├── user/
│   ├── .abstract.md          # L0: Content summary
│   ├── .overview.md          # User profile
│   └── memories/             # User memory storage
│       ├── .overview.md      # Memory overview
│       ├── preferences/      # User preferences
│       ├── entities/         # Entity memories
│       └── events/           # Event records
│
├── agent/
│   ├── .abstract.md          # L0: Content summary
│   ├── .overview.md          # Agent overview
│   ├── memories/             # Agent learning memories
│   │   ├── .overview.md
│   │   ├── cases/            # Cases
│   │   └── patterns/         # Patterns
│   ├── instructions/         # Agent instructions
│   └── skills/               # Skills directory
│
└── resources/{project}/      # Resource workspace
```

## URI Examples

### Resources

```
wfs://resources/                           # All resources
wfs://resources/my-project/                # Project root
wfs://resources/my-project/docs/           # Docs directory
wfs://resources/my-project/docs/api.md     # Specific file
```

### User Data

```
wfs://user/                                # User root
wfs://user/memories/                       # All user memories
wfs://user/memories/preferences/           # User preferences
wfs://user/memories/preferences/coding     # Specific preference
wfs://user/memories/entities/              # Entity memories
wfs://user/memories/events/                # Event memories
```

### Agent Data

```
wfs://agent/                               # Agent root
wfs://agent/skills/                        # All skills
wfs://agent/skills/search-web              # Specific skill
wfs://agent/memories/                      # Agent memories
wfs://agent/memories/cases/                # Learned cases
wfs://agent/memories/patterns/             # Learned patterns
wfs://agent/instructions/                  # Agent instructions
```

The short `wfs://user/...` and `wfs://agent/...` forms above are
relative to the current request identity. OpenViking expands them internally to
explicit namespace paths such as `wfs://user/{user_id}/...` and
`wfs://agent/{agent_id}/...` before storage and retrieval.

### Session Data

```
wfs://session/{session_id}/                # Session root
wfs://session/{session_id}/messages/       # Session messages
wfs://session/{session_id}/tools/          # Tool executions
wfs://session/{session_id}/history/        # Archived history
```

## Path Variables

Viking URI supports path variables for dynamic path generation. This is especially useful for organizing time-series data like emails, logs, daily reports, etc.

### Variable Syntax

```
{namespace:key}
```

- **namespace**: Variable provider namespace (e.g., `calendar`, `env`, `user`)
- **key**: Variable name within the namespace

### Calendar Variables

The `calendar` namespace provides date-related variables:

| Variable | Description | Example (2026-05-07) |
|----------|-------------|----------------------|
| `{calendar:today}` | Full date path | `2026/05/07` |
| `{calendar:yesterday}` | Yesterday's date path | `2026/05/06` |
| `{calendar:tomorrow}` | Tomorrow's date path | `2026/05/08` |
| `{calendar:year}` | Year | `2026` |
| `{calendar:month}` | Month with leading zero | `05` |
| `{calendar:day}` | Day with leading zero | `07` |
| `{calendar:ym}` | Year/month | `2026/05` |
| `{calendar:quarter}` | Quarter (Q1-Q4) | `Q2` |
| `{calendar:yq}` | Year/quarter | `2026/Q2` |
| `{calendar:week}` | ISO week number with leading zero | `18` |
| `{calendar:yw}` | Year/ISO week | `2026/w18` |

### Usage Examples

```python
# Organize emails by date
wfs://resources/emails/{calendar:today}/inbox
# Renders to: wfs://resources/emails/2026/05/07/inbox

# View yesterday's logs
wfs://resources/logs/{calendar:yesterday}/app.log
# Renders to: wfs://resources/logs/2026/05/06/app.log

# Pre-upload tomorrow's tasks
wfs://resources/tasks/{calendar:tomorrow}/todo.md
# Renders to: wfs://resources/tasks/2026/05/08/todo.md

# Monthly logs
wfs://resources/logs/{calendar:year}/{calendar:month}/app.log
# Renders to: wfs://resources/logs/2026/05/app.log

# Daily snapshots
wfs://resources/snapshots/{calendar:today}/
# Renders to: wfs://resources/snapshots/2026/05/07/
```

### Resolution

Path variables are resolved **server-side** at the time of API execution. The CLI/SDK passes the URI template as-is, and the server renders it to a concrete path based on the current context (time, authenticated user, etc.).

### Use with CLI

```bash
# Add today's emails, --parent-auto-create can be shortened to -p
ov add-resource --parent-auto-create "wfs://resources/emails/{calendar:today}/inbox" ./emails/*.eml

# Read yesterday's log
ov read "wfs://resources/logs/{calendar:yesterday}/app.log"

# Prep tomorrow's tasks
ov write --uri "wfs://resources/tasks/{calendar:tomorrow}/todo.md" --content "Plan the day"

# Upload monthly report, --parent-auto-create can be shortened to -p
ov add-resource --parent-auto-create "wfs://resources/reports/{calendar:ym}" ./report.pdf
```

## Directory Structure

```
wfs://
├── resources/       # Independent resources
│   └── {project}/
│       ├── .abstract.md
│       ├── .overview.md
│       └── {files...}
│
├── user/{user_id}/
│   ├── profile.md                # User basic info
│   └── memories/
│       ├── preferences/          # By topic
│       ├── entities/             # Each independent
│       └── events/               # Each independent
│
├── agent/{agent_id}/             # Agent root when isolate_agent_scope_by_user = false
│   ├── skills/                   # Skill definitions
│   ├── memories/
│   │   ├── cases/
│   │   └── patterns/
│   ├── workspaces/
│   └── instructions/
│
├── agent/{agent_id}/user/{user_id}/   # Agent root when isolate_agent_scope_by_user = true
│   ├── skills/
│   ├── memories/
│   ├── workspaces/
│   └── instructions/
│
└── session/{user_space}/{session_id}/
    ├── messages/
    ├── tools/
    └── history/
```

Agent namespace shape is controlled by per-account namespace policy:

- `isolate_agent_scope_by_user = false`: `wfs://agent/{agent_id}/...`
- `isolate_agent_scope_by_user = true`: `wfs://agent/{agent_id}/user/{user_id}/...`

`memory.agent_scope_mode` is deprecated and ignored.

## URI Operations

### Parsing

```python
from openviking_cli.utils.uri import VikingURI

uri = VikingURI("wfs://resources/docs/api")
print(uri.scope)      # "resources"
print(uri.full_path)  # "resources/docs/api"
```

### Building

```python
# Join paths
base = "wfs://resources/docs/"
full = VikingURI(base).join("api.md").uri  # wfs://resources/docs/api.md

# Parent directory
uri = "wfs://resources/docs/api.md"
parent = VikingURI(uri).parent.uri  # wfs://resources/docs
```

## API Usage

### Targeting Specific Scopes

```python
# Search only in resources
results = client.find(
    "authentication",
    target_uri="wfs://resources/"
)

# Search only in user memories
results = client.find(
    "coding preferences",
    target_uri="wfs://user/memories/"
)

# Search only in skills
results = client.find(
    "web search",
    target_uri="wfs://agent/skills/"
)
```

### File System Operations

```python
# List directory
entries = await client.ls("wfs://resources/")

# Read file
content = await client.read("wfs://resources/docs/api.md")

# Get abstract
abstract = await client.abstract("wfs://resources/docs/")

# Get overview
overview = await client.overview("wfs://resources/docs/")
```

## Special Files

Each directory may contain special files:

| File | Purpose |
|------|---------|
| `.abstract.md` | L0 abstract (~100 tokens) |
| `.overview.md` | L1 overview (~2k tokens) |
| `.relations.json` | Related resources |
| `.meta.json` | Metadata |

## Best Practices

### Use Trailing Slash for Directories

```python
# Directory
"wfs://resources/docs/"

# File
"wfs://resources/docs/api.md"
```

### Scope-Specific Operations

```python
# Add resources only to resources scope
await client.add_resource(url, to="wfs://resources/project/")

# Skills go to agent scope
await client.add_skill(skill)  # Automatically to wfs://agent/skills/
```

## Related Documents

- [Architecture Overview](./01-architecture.md) - System architecture
- [Context Types](./02-context-types.md) - Three types of context
- [Context Layers](./03-context-layers.md) - L0/L1/L2 model
- [Storage Architecture](./05-storage.md) - VikingFS and AGFS
- [Session Management](./08-session.md) - Session storage structure
