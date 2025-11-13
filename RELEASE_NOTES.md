# Release Notes

## Version 0.2.0 - Claude Code Edition (2024-11-12)

### 🎉 Major Enhancement Release

This release transforms Erosolar into a **comprehensive, Claude Code-like CLI agent** with all the tools and capabilities needed for professional software development workflows.

#### 📝 NEW: Advanced File Operations (Claude Code-like)
- **`edit_file`**: Precise find/replace editing without rewriting entire files
- **`glob_files`**: Pattern-based file discovery (e.g., `**/*.py`, `src/**/*.ts`)
- **`grep_files`**: Powerful regex search across files with ripgrep-like functionality

#### 🔧 NEW: Git Integration
- **`git_status`**: Show working tree changes
- **`git_diff`**: View actual code changes (staged/unstaged)
- **`git_commit`**: Create commits with automatic staging
- **`git_log`**: View commit history

#### 📝 NEW: Task & Workflow Management
- **`manage_todos`**: Built-in todo/task tracking with pending/in_progress/completed states

#### 🧪 NEW: Jupyter Notebook Support
- **`edit_notebook_cell`**: Programmatically modify .ipynb files

#### 📊 NEW: Code Analysis & Quality
- **`analyze_code_quality`**: Comprehensive code metrics and issue detection for Python, JavaScript, TypeScript, Java, Go, Rust, C/C++

#### 🌐 NEW: Enhanced Web Operations
- **`web_search_simple`**: Quick DuckDuckGo searches

#### 🎯 Total Tool Count: 26+ tools across 11 categories

**New Documentation:**
- **TOOLSET.md**: Complete reference guide with examples for all tools
- **Enhanced README**: Comprehensive workflow examples

**Migration:** `pip install --upgrade erosolar` - No breaking changes!

---

## Version 0.1.0 - Initial Release (2024-01-XX)

## 🎉 Initial Release - Erosolar Universal Agent

Erosolar is a powerful universal LangGraph agent CLI that brings advanced AI capabilities to your terminal with MCP support, persistent tools, and research memory.

### ✨ Key Features

#### 🧠 **Intelligent Agent Core**
- **LangGraph-based planner/executor loop** powered by LangChain + DeepSeek Reasoner
- **Smart tool selection** with semantic search and BM25 retrieval
- **Interactive terminal UI** with rich formatting and streaming responses

#### 🛠️ **Built-in Research & Automation Tools**
- **Tavily search & extract** - Web research with AI-powered extraction
- **HTTP fetch** - Retrieve and analyze web content
- **Weather data** - Current conditions via wttr.in
- **Shell & Python execution** - Run commands and scripts on-demand
- **Headless browser** - Playwright-powered web automation

#### 💾 **Persistent Tool Library**
- **Create custom tools** - Build reusable shell or Python tools that persist across sessions
- **SQLite-backed storage** - All tools stored in local database
- **Full CRUD operations** - Create, update, run, list, and delete tools
- **Structured arguments** - Define JSON schemas for tool parameters

#### 📚 **Research Vault**
- **Long-term memory** - Store research notes that survive across multiple sessions
- **Namespace organization** - Group notes by project or topic
- **Append mode** - Build up investigations over time
- **Metadata support** - Attach structured metadata to notes

#### 🔌 **MCP Integration**
- **Model Context Protocol bridge** - Connect to MCP servers
- **Dynamic tool loading** - MCP tools appear alongside built-in capabilities
- **Flexible configuration** - JSON-based server configuration

#### 🌐 **REST API**
- **HTTP endpoint** at `http://127.0.0.1:9000/chat`
- **Programmatic access** - Send prompts via POST requests
- **Integration-friendly** - Connect from other applications

### 📦 Installation

```bash
pip install erosolar
```

### 🚀 Quick Start

```bash
# Set required API keys
export DEEPSEEK_API_KEY="your-deepseek-key"
export TAVILY_API_KEY="your-tavily-key"

# Optional: Configure MCP servers
export AGENT_MCP_SERVERS='{"server_name": {"command": "...", "args": [...]}}'

# Launch the agent
erosolar --verbose
```

### 📝 Usage Examples

**Using the Tool Library:**
```python
# Create a custom tool
tool_library(action='create', name='git_status',
             description='Check git repository status',
             kind='shell', body='git status --short')

# Run your custom tool
tool_library(action='run', name='git_status')
```

**Using the Research Vault:**
```python
# Store research notes
research_vault(action='set', namespace='project-alpha',
               key='hypothesis', content='Initial findings...')

# Retrieve notes later
research_vault(action='get', namespace='project-alpha', key='hypothesis')
```

### ⚙️ Configuration

**Storage Location:**
- Default: `~/.universal_agent/agent_state.sqlite3`
- Override with `AGENT_STATE_DIR` or `AGENT_STATE_DB`

**MCP Servers:**
- Configure via `AGENT_MCP_SERVERS` environment variable (JSON)
- Or create `mcp_servers.json` in working directory
- See `mcp_servers.sample.json` for examples

### 🔗 Links

- **PyPI**: https://pypi.org/project/erosolar/
- **GitHub**: https://github.com/dragonghidra/erosolar
- **Issues**: https://github.com/dragonghidra/erosolar/issues

### 📋 Requirements

- Python >= 3.10
- DeepSeek API key (for reasoning)
- Tavily API key (for web research)

### 🙏 Acknowledgments

Built with:
- LangGraph & LangChain for agent orchestration
- DeepSeek for advanced reasoning capabilities
- Tavily for web research
- Model Context Protocol for extensibility

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
