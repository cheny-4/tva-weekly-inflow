# Skill authoring kit

Use these framework-neutral documents to wrap the TVA weekly inflow baseline in your own agent or tool framework. The installable skill is separately available at [`../skills/tva-weekly-inflow/SKILL.md`](../skills/tva-weekly-inflow/SKILL.md).

- [Capabilities and permissions](CAPABILITIES.md): what the pipeline can do and which actions spend compute or change state.
- [Data contract](DATA_CONTRACT.md): basin support, features, units, masks, site identity and forecast limitations.
- [Interface and registry](registry.json): machine-readable command templates, side effects and output files. Execute argument arrays rather than interpolating untrusted text into shell commands.
- [Weekly forcing schema](weekly-forcings.schema.json): JSON representation of the CSV input contract and its required metadata.

The executable interface is `python pipeline.py`. These documents describe the verified baseline, not an orchestration service. An agent must have a local clone with its artifacts and dependencies, or authorized access to that clone on Frontier. The registry does not contain credentials, delegate access, authorize a training job, or automatically process raw maps.

Integration example: resolve the repository path; validate the forcing metadata and known site IDs; run `inspect`; invoke `infer` into a new output directory; read provenance and report invalid windows. For a figure request, use existing paired predictions and `plot` instead. See the skill references for detailed commands.
