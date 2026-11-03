# Simulation visualization provenance

The paper's three displayed objects are selected examples, not a random estimate of grasp success. They are cellphone (frozen and full-weight GRPO fail; residual GRPO and scaled shift succeed), bottle 2 (all four fail), and bottle 3 (all four succeed). S/F labels are the simulator evaluation's success flags. Every scene is an Isaac Gym simulation render.

The source visualization release re-rendered the first evaluation episode of the selected checkpoint's first training run, using seed 4242 and the 141 seen-category objects. Seven of nine policy–condition evaluations reproduce every archived first-episode flag. The raw output shift differs on 10 objects and nominal residual GRPO on 7 objects; the cause was not established. All objects displayed in the figures and videos match the archived flags. Paper statistics use the archived evaluations, not this re-render.

The source release checked 4,518 used frames against server checksums. Presentation involved cropping, uniform scaling, annotations and video encoding; object, grasp and outcome content was not altered. The v14 revision preserves the existing figure-image bytes and performs no new rendering or simulation.

Source records: DexGrasp_仿真可视化_1009_v2/README_可视化_v2.md, Captions_and_notes_EN.txt, check.json, MANIFEST.json and frames_checked.json, retained with the research archive. The detailed provenance is recorded here so the figure caption can focus on interpreting the displayed scenes.
