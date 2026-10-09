UNetMamba FloodPlanet Final Handoff
===================================

Contents:
- UNetMamba_Final_Handoff.md: complete factual handoff of the implemented and evaluated UNetMamba baseline, actual settings/results, methodological caveats, reviewed literature and extension candidates.
- research_sources/flood_literature_review_2020_2026.xlsx: prior literature matrix.
- research_sources/flood_literature_review_2024_2026.xlsx: recent literature matrix and research-gap summary.
- research_sources/Research_for_flood_inundation.docx: detailed research notes including DeepSARFlood and extension candidate notes.

This handoff bundle does NOT contain the FloodPlanet dataset or the trained .pt checkpoint. The actual model checkpoint is in the separately downloaded UNetMamba_FloodPlanet_Final_Package.zip. That final package is not self-contained model source: it does not include unetmamba/src/ or the full selective-scan kernel source. Use the project GitHub source for those files and reapply the documented compiler compatibility patch.
