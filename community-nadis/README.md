# Community Nadis

Nadis built by the community that extend the Nadiru engine.

## What's a Nadi?

A Nadi is any application that connects to a running Nadiru engine. It uses the same three endpoints as everything else — `/connect`, `/generate`, `/query`. No special access, no private APIs.

## Contributing a Nadi

1. Create a directory named `your-nadi-name/`
2. Include a `README.md` explaining what it does and how to run it
3. Include all source code and a `requirements.txt` if needed
4. Open a PR

Each Nadi should be self-contained. Don't modify engine code.

## Ideas

- **Cost dashboard** — query `/query` and visualize spend by provider, model, time period
- **Model health monitor** — watch for error patterns and surface degraded providers
- **Routing debugger** — show why the Conductor made specific routing decisions
- **CLI chat** — simple terminal interface to the engine
- **Discord/Slack bot** — connect your chat platform to the engine
- **Batch processor** — send bulk prompts with cost controls
