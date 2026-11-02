# Agent integration kit

Use the ready skill at ../skills/tva-weekly-inflow/SKILL.md or build an integration around registry.json, DATA_CONTRACT.md, CAPABILITIES.md and weekly-forcings.schema.json. The supported executable interface is ../pipeline.py; models.json names the two frozen versions. Select --model explicitly and keep its checkpoint, runoff definition, static descriptors and scalers together. Both versions have bias correction disabled.

HEADWATER map ingestion is implemented by the preprocess capability. See ../docs/headwater_preprocessing.md and the runnable synthetic example generator. Preserve the packaged spatial support, original masks and chronological splits. Actual future HEADWATER schema/forecast skill still requires validation against real files.

The schema documents the JSON equivalent of CSV+metadata; runtime additionally validates chronology, existing heads, window completeness, model/feature compatibility and spatial coverage. The skill references the checkout for its scripts, model data and spatial files. Access to this private repository remains necessary to clone it; no credential or automatic permission expansion is provided.
