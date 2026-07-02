---
name: ov-search-context
description: Search context data(memories, skills and resource) from OpenViking Context Database (aka. ov). Trigger this tool when 1. need information that might be stored as memories, skills or resources on OpenViking; 2. is explicitly requested searching files or knowledge; 3. sees `search context`, `search openviking`, `search ov` request.
compatibility: CLI configured at `~/.openviking/ovcli.conf`
---
# OpenViking (OV) context searching
The `ov search` command performs context-aware retrieval across all memories and resources in OpenViking — combining semantic understanding with directory recursive retrieval to find the most relevant context for any query.

## Table of Content
- When to Use
- Sub-commands for search
  - List directories (`ov ls`)
  - Tree view (`ov tree`)
  - Semantic Search (`ov find`)
  - Content Pattern Search (`ov grep`)
  - File Glob Search (`ov glob`)
  - Full content read (`ov read`)
  - Get overview (`ov overview`)
  - Get Abstract (`ov abstract`)
- Prerequisite

## When to Use

- Finding specific information within imported resources or saved memories
- Retrieving context about topics, APIs, or patterns previously added
- Searching across project documentation, code, and learnings
- When an agent needs to reference previously stored knowledge

> note: cli command can be outdated, when sees error, use `--help` to get latest usage

## Sub-commands for search

### List Contents (`ov ls`)

Browse directory structure:

```bash
# List root directory
ov ls

# List specific directory
ov ls wfs://resources/my-project/docs/

# Simple path output (only uris, no metadata)
ov ls wfs://resources --simple

# Show hidden files
ov ls wfs://resources --all

# Control output limits (default 256)
ov ls wfs://resources --node-limit 50

# Control abstract info length limit for each node (default 256)
ov ls wfs://resources --abs-limit 128
```

### Tree View (`ov tree`)

Visualize directory hierarchy:

```bash
# Show tree structure
ov tree wfs://resources

# Control depth limits (default 3)
ov tree wfs://resources --level-limit 2

# Control node limits
ov tree wfs://resources --node-limit 100 --abs-limit 128

# Show all files including hidden
ov tree wfs://resources --all
```

### Semantic find (`ov find`)

Find method with semantic relevance ranking:

```bash
# Basic find across all context
ov find "how to handle API rate limits"

# Find within specific URI scope
ov find "authentication flow" --uri "wfs://resources/my-project"

# Limit results and set relevance score threshold
ov find "error handling" --node-limit 5 --threshold 0.3
```

### Content Pattern Search (`ov grep`)

Literal pattern matching:

```bash
# Find exact text pattern (Note: this is expensive, and suggest within specific small URI scope)
ov grep "wfs://resources" "TODO:" --uri "wfs://resources/my-project"

# Case-insensitive search
ov grep "wfs://resources" "API_KEY" --ignore-case --uri "wfs://resources/my-project"

# Limit results and set node limit
ov grep "wfs://resources" "API_KEY" --node-limit 5 --uri "wfs://resources/my-project"
```

### File Glob Search (`ov glob`)

File path pattern matching:

```bash
# Find all markdown files (Note: this is expensive, and suggest within specific small URI scope)
ov glob "**/*.md" --uri "wfs://resources/my-project"

# Limit results and set node limit
ov glob "**/*.md" --uri "wfs://resources/my-project" --node-limit 5
```

### Read File Content (`ov read`)

Retrieve full content (L0-L2 layer):

```bash
# Read full content
ov read wfs://resources/docs/api/api-1.md

# Read first 10 lines of api-2.md
ov read wfs://resources/docs/api/api-2.md | head -n 10

# Read abstract (L0 - quick summary)
ov abstract wfs://resources/docs/api/
ov read wfs://resources/docs/api/.abstract.md

# Read overview (L1 - key points)
ov overview wfs://resources/docs/api/
ov read wfs://resources/docs/api/.overview.md
```

### Combining Search

Use search results to guide further actions:

```bash
ov ls wfs://resources/

# Search for relevant files
ov search "authentication" --uri "wfs://resources/project-A"

# Get overview for context
ov overview wfs://resources/project-A/backend

# Decide to read specific content
ov read wfs://resources/project-A/backend/auth.md
```

## Prerequisites

- CLI configured: `~/.openviking/ovcli.conf`
- Resources or memories previously added to OpenViking
