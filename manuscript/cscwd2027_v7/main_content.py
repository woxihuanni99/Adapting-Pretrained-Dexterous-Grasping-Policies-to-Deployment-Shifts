title='Recovery–Retention Trade-offs in Adapting a Pretrained Dexterous Grasp Planner'
q=p(title,'paper title');q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.space_after=Pt(10)
authors=D.add_table(rows=1,cols=2);authors.autofit=False
authorlines=[['Haoze Lan','School of Intelligent Manufacturing','(Pen-Tung Sah Institute of Micro-Nano','Science and Technology), Xiamen University','Xiamen 361100, China'],
             ['Yongbin Yuan','Fuzhou University','Fuzhou, China','yuanyongbin@fzu.edu.cn','Corresponding author']]
for cell,lines in zip(authors.rows[0].cells,authorlines):
    cell.width=Inches(3.62);cell.text=''
    for i,line in enumerate(lines):
        q=cell.paragraphs[0] if i==0 else cell.add_paragraph();q.style=D.styles['Affiliation'];q.alignment=WD_ALIGN_PARAGRAPH.CENTER
        q.paragraph_format.space_before=Pt(0);q.paragraph_format.space_after=Pt(0);q.paragraph_format.line_spacing=1
        rr=q.add_run(line);rr.font.size=Pt(11);rr.italic=i>0
sec=D.add_section(WD_SECTION_START.CONTINUOUS)
cols=sec._sectPr.find(qn('w:cols'));cols.set(qn('w:num'),'2');cols.set(qn('w:space'),'360')
q=D.paragraphs[-1];q.paragraph_format.space_after=Pt(7);q.paragraph_format.space_before=Pt(0);q.paragraph_format.line_spacing=Pt(1);q.add_run().font.size=Pt(1)

z=S['interpolation'];gap=z['estimate']['effect']
p(f'Abstract—Adapting a pretrained grasp planner requires recovering from faults while retaining normal grasping performance. We compare full-weight and frozen-planner residual updates, each optimized with proximal policy optimization (PPO) or group relative policy optimization (GRPO), in Isaac Gym using Shadow and Allegro hands. At the reference little-finger lock, residual procedures recover more successfully at comparable observed nominal costs: their test curves have a {gap:.1f}-percentage-point interpolated mean separation over the common loss range. Under a shared development selection rule, refining the full-weight learning-rate grid leaves a 28.0-point recovery advantage with 3.1 points of additional nominal loss. Changing the lock to 75% of the joint ranges reduces the estimated contrast to 0.4 points. An extracted constant correction achieves 66.7% success, compared with 64.2% for its source residual network. Compact corrections thus offer strong recovery for the reference fault, while posture tests identify the range over which that recovery persists.','Abstract')
p('Keywords—dexterous grasping, policy adaptation, reinforcement learning, recovery, retention','Keywords')

h('I. INTRODUCTION')
p('A grasping cell must often adapt an existing policy after a sensor or actuator changes. The engineering choice is which parts of the policy to update and how much normal performance to spend on recovery. Pretrained dexterous policies handle diverse objects {{162,157}}; DemoGrasp generates an open-loop grasp from one planning decision {{164}}. This setting makes the cost of updating a working planner directly measurable.')
p('Full-weight fine-tuning changes the pretrained network {{152}}, whereas residual learning adds a correction to a frozen planner {{131,140,166}}. A large fault-recovery difference can reflect both the update procedure and the nominal-performance operating point chosen by tuning. Comparing success alone obscures that distinction. Training variability and configuration selection also affect the measured separation {{127,136,151}}.')
p('We study three questions: how recovery changes with nominal loss, how the comparison changes with fault posture, and how complex the correction must be. Four procedures receive about 19,000 grasp attempts each. One development rule selects a learning rate jointly on little-finger lock and point-cloud noise; additional tests cover calibration shifts, combined faults and a second hand.')
p('Our contributions are: (1) a reproducible recovery–retention comparison that separates observed curve differences from development-selected outcomes; (2) a lock-posture boundary that persists after full-weight grid refinement and same-server reevaluation; and (3) compact-correction diagnostics linking successful reference-lock recovery to a concentrated output shift. The resulting measurements support concrete update choices for a simulated grasping cell.')

h('II. RELATED WORK')
p('Dexterous grasping. UniDexGrasp and UniDexGrasp++ learn broad grasping behaviors {{162,157}}. Dexterous Grasp Transformer predicts grasp configurations from point clouds {{161}}, and CrossDex studies learning across hands {{165}}. DemoGrasp edits a demonstration with a one-step policy {{164}}, providing the planner studied here.')
p('Policy adaptation. Residual policy learning adds a learned correction to an existing controller {{169,170}}. RESPRECT and ResDex apply this idea to dexterous grasping {{131,140}}; Policy Decorator refines pretrained policies online {{166}}. DPPO updates policy weights {{152}}, while DSRL and RFS steer frozen generative policies in latent coordinates {{156,155}}. LoRA and task-specific adapters provide other compact update families {{138,144}}. Robot damage adaptation has also searched simulated behavior repertoires {{174}}, and RMA combines a base policy with an adaptation module {{175}}. Residual Fault Adaptation uses recurrent control and execution history for joint faults during in-hand manipulation {{134}}. Our comparison uses the released one-step observation and execution interface.')
p('Retention during fine-tuning. EWC protects weights important to previous tasks {{173}}. LP-FT studies feature distortion during adaptation {{171}}; WiSE-FT combines pretrained and fine-tuned weights to preserve robustness {{172}}. Forgetting mitigation and calibrated offline-to-online learning also address the cost of updating pretrained behavior {{159,149}}. Here, the reported operating points expose that cost alongside fault recovery, using independent training runs for inference {{127,136,151}}.')

h('III. ADAPTATION DESIGN')
sub('A. Planner, observations and shifts')
p('We use released Shadow and Allegro DemoGrasp checkpoints in Isaac Gym {{164,146}}. The observation contains the initial end-effector pose, object pose and point cloud; joint state and fault labels are absent. A policy selects one grasp edit, which executes open loop and receives a binary lift outcome. At the policy-decision level this is a one-step contextual-bandit problem. Shadow actions have 24 dimensions: three wrist translations, three wrist rotations and 18 hand-pose edits. The Gaussian planner samples a pre-squashing action u and executes a = tanh(u):')
equation(['u = ',('sub','m','0'),'(s) + ',('sub','σ','0'),' ⊙ ε,   ε ∼ N(0, I).'],1)
p('Here m₀ and σ₀ are the pretrained mean and sampling scale, and ⊙ denotes elementwise multiplication. Reference locks hold joint targets from the first demonstration frame. Alternative locks set each controlled joint to 25% or 75% of its own range. Each experiment selects a shift-specific model before the grasp; the model itself does not infer the fault.')
p('The six Shadow conditions are little-finger lock (LF), thumb lock (TH), 1 cm Gaussian noise per point-cloud coordinate (PCL), a vertical offset U(−3, 3) cm (ZR), a fixed 2 cm offset plus PCL (Z2+PCL), and TH+PCL. PCL changes the point cloud while leaving the separate object-pose observation unchanged. Allegro uses TH and Z2. Training uses 768 objects. A grasp fails if a palm or fingertip keypoint crosses below the tabletop. The primary setting retains the released collision filter, disabling robot–table contact in half the environments; a separate check enables all robot–table collisions and adds a ground plane.')

sub('B. Four update procedures')
p('Full-weight PPO {{153}} updates the planner and scale; its value loss (coefficient 2.0) also updates the shared PointNet backbone. Full-weight GRPO {{154}} updates the backbone and mean head, fixes the scale to max(σ₀, 0.05), and normalizes rewards within groups of eight actions. Residual procedures freeze the planner and sample')
equation(['u = ',('sub','m','0'),'(s) + ',('sub','σ','0'),' ⊙ [',('sub','δ','ϕ'),'(s) + ',('sup','e','ρ'),' ⊙ ε].'],2)
p('The correction network uses a layer-normalized MLP with two 256-unit hidden layers. Its input concatenates the non-point-cloud observation and frozen PointNet embedding, which we call planner features. Zero initialization of its output layer and scale vector ρ reproduces the pretrained policy. Residual PPO jointly optimizes its correction and value networks with one Adam and gradient clipping (value-loss coefficient 0.5); residual GRPO uses the same 64-state × eight-action grouping as full-weight GRPO. Table I records optimizer calls as well as interactions.')
u=S['optimizer_calls']
table('TABLE I. UPDATE CONFIGURATIONS', ['Procedure','Scale','Epochs','Adam\nsteps','Attempts'],
      [[r['method'],['Learned','Fixed','Learned','Learned'][i],r['epochs'],r['adam_steps'],f"{r['attempts']:,}"] for i,r in enumerate(u[:4])],
      [1.03,.55,.43,.53,.94],
      'Fixed scale: max(σ₀, 0.05). A phase uses a fixed object list and passes its checkpoint forward. Full-weight PPO uses three phases/77 iterations; full-weight GRPO uses 12/38; residual procedures use 12/37. Adam calls = iterations × epochs × minibatches (two for full-weight PPO, four otherwise).')

sub('C. Selection, evaluation and uncertainty')
p('The development set has 64 objects disjoint from training and test instances. Each procedure uses one learning rate for LF and PCL. Original full-weight rates are {10⁻⁵, 3×10⁻⁵, 10⁻⁴}; residual rates are {10⁻⁴, 3×10⁻⁴, 10⁻³, 3×10⁻³, 10⁻²}. A rate is feasible if mean nominal loss is at most five percentage points (pp) in both conditions. Feasible rates maximize average fault success; ties favor smaller worst loss and then a smaller rate. Three initial runs expand once to five when either development condition lies within one point of the boundary. Selection uses unrounded values from 512 fault and 2,048 nominal attempts per model. Original selected rates are 10⁻⁵ for both full-weight procedures, 3×10⁻³ for residual PPO and 10⁻³ for residual GRPO; the other conditions reuse these rates with separately trained models.')
p('Evaluation uses the policy mean and seed 4242. A run averages seen-category success (141 instances, 1,128 attempts) and unseen-category success (78 instances represented in 100 slots, 800 attempts). Twenty-two unseen instances occur twice in that list. Nominal loss is frozen nominal success minus adapted nominal success. Development feasibility and achieved test loss are reported separately.')
p('The original protocol fixed configurations and directional tests after exploratory use of the test objects; LF/PCL evaluations reuse selection runs. Training runs are the statistical units. Selected-configuration contrasts use two-sided 95% Welch intervals; the original nine directional tests use one-sided p-values with Holm correction (Table IV). Residual-minus-full-weight contrasts equally weight PPO and GRPO; optimizer contrasts average over update types. Individual operating-point intervals use Student t across runs, conditional on the fixed frozen reference. Curve analysis linearly interpolates observed test means and averages the equal-optimizer difference uniformly over their common nominal-loss range. A percentile bootstrap uses 10,000 draws (seed 110205), resampling paired fault/nominal scores within each rate and recomputing common support. This descriptive analysis is separate from the nine-test family.')

h('IV. RECOVERY AND RETENTION')
sub('A. Observed operating points')
orig=json.loads((R.parent/'cscwd2027_v4/table_audit.json').read_text(encoding='utf8'))
oldtable=orig['TABLE II. ORIGINAL SUCCESS (%) AND NOMINAL LOSS (pp)']
table('TABLE II. ORIGINAL OPERATING POINTS',oldtable['headers'],oldtable['rows'],[.64,.4,.61,.61,.61,.61],
      'Success (%) with nominal loss (pp) in parentheses. A: Allegro; otherwise Shadow. Frozen nominal success is 91.0%/86.8% for Shadow/Allegro. n = 3, except residual PPO LF/PCL (n = 5).')
p('Reference LF is the strongest separation. The frozen planner achieves 16.9% success; full-weight procedures reach about 32% and residual procedures 64–65%. Full-weight means are slightly higher in the other seven conditions. Six of eight adapted Allegro means exceed five points of nominal loss. PPO–GRPO contrasts range from −0.3 to +1.8 pp. The one-step decision structure provides a setting in which both optimizers learn from immediate binary outcomes.')
figure('fig3_sim_examples.png','Fig. 1. Isaac Gym simulation renders at reference LF. Three deliberately selected objects, not a random sample; S/F is the recorded success flag. The scaled shift is the 24-parameter mean correction in Section V. Frames use the first training run and first seen-category episode. Two of nine visualization reruns differed from archived outcomes; all displayed objects exclude those discrepancies.')
p(f'Comparing the curves puts recovery on a common nominal-cost axis (Fig. 2). Over their shared {z["estimate"]["lower"]:.1f}–{z["estimate"]["upper"]:.1f} pp loss range, the equally weighted PPO/GRPO curve separation averages {gap:.1f} pp (95% bootstrap interval [{z["percentile_CI95"][0]:.1f}, {z["percentile_CI95"][1]:.1f}]). An observed PPO pair illustrates the result: full-weight PPO reaches 50.2% success at 5.1 pp loss, versus 65.2% at 5.4 pp for residual PPO. Interpolation provides a descriptive summary of the observed operating points.')
figure('fig1_tradeoff.png','Fig. 2. Original reference-LF recovery–retention curves. Shading marks the loss range shared by all four curves. Small markers show training runs; stars show original development choices. The main panel enlarges the low-loss region; the inset shows the full range. Fourteen of 16 original candidates have plotted test records; both residual 10⁻² candidates are omitted.')

sub('B. Grid refinement and achieved retention')
p('Under the original shared development rule, the averaged residual-minus-full-weight contrast is 32.7 [29.4, 36.1] pp with 4.1 pp additional nominal loss. Full-weight PPO at 3×10⁻⁵ misses feasibility by 0.20 pp on development PCL, while full-weight GRPO at that rate loses 8.45 pp on development LF. These choices put the methods at different achieved test costs.')
p('We added 2×10⁻⁵ to both full-weight grids and completed 22 training runs, including the prescribed GRPO additions. All selected models and frozen references were then evaluated on the same server. Full-weight PPO selects 2×10⁻⁵, GRPO keeps 10⁻⁵, and residual rates retain their original choices. Forty archived development checkpoints show 1.20 pp mean loss drift with no original-grid feasibility change; the new PPO choice comes from the added rate. Table III records recovery and retention uncertainty.')
labels={'fwppo':'Full PPO','fwgrpo':'Full GRPO','res_ppo':'Residual PPO','res_grpo':'Residual GRPO'}
table('TABLE III. SAME-SERVER UPDATE RECORD', ['Procedure (n)','LF\nsuccess','LF loss\n[95% CI]','PCL success\n(loss)'],
      [[labels[k]+f" ({sel(k)['loss']['n']})",f"{sel(k)['success']['mean']:.1f}",statfmt(sel(k)['loss']),f"{sel(k,'pn10')['success']['mean']:.1f} ({sel(k,'pn10')['loss']['mean']:.1f})"] for k in labels],
      [.94,.47,1.18,.89],
      'Success in %, loss in pp; frozen nominal success 91.3%. Rates: 2×10⁻⁵, 10⁻⁵, 3×10⁻³ and 10⁻³, in row order. Intervals quantify variation across training runs. Five pp is the development selection limit; the columns report achieved test costs.')
p('Grid refinement leaves a 28.0 [23.1, 33.0] pp reference-LF contrast and 3.1 pp additional nominal loss. The PPO contrast is 24.1 [14.2, 33.9] pp. Under PCL, residual-minus-full-weight is −2.7 [−5.1, −0.4] pp. The reference-lock recovery gain therefore persists after the grid extension, while its magnitude depends on the selected operating points.')
p('A fixed-scale, two-epoch residual GRPO control reaches 68.0% LF success at 3×10⁻⁴, but its development loss is 7.8 pp and it fails the shared rule. The newly tested full-weight GRPO 2×10⁻⁵ also fails, with 5.6 pp development PCL loss.')

sub('C. Lock posture changes the useful correction')
p('Reference-trained models retain their large recovery difference at the 25% lock. At 75%, the same-server contrast is 0.4 [−2.7, 3.5] pp, compared with 28.0 pp at reference LF (Fig. 3). Frozen success remains about 16–19% across these postures. A direct original-protocol within-run comparison gives a 32.9 [27.6, 38.3] pp reduction in the procedure contrast from reference to 75%.')
figure('fig2_posture.png','Fig. 3. Residual-minus-full-weight success and two-sided 95% intervals. Circles: original selection; squares: expanded full-weight selection and original residual choices, all reevaluated on the same server. The first three rows reuse reference-trained models; nominal evaluation is unchanged across those postures. Lower rows use separate retrained models. Right-hand frozen values use the original protocol.')
p('Retraining at 75% yields roughly 41–43% success for all four procedures, with an averaged contrast of −1.5 [−4.8, 1.8] pp. With ring and little fingers locked together, the residual contrast is 7.4 [2.5, 12.4] pp (five runs, including the initial three). The relative recovery performance changes with lock geometry.')

sub('D. Retention controls and directional tests')
p('A KL-regularized full-weight GRPO baseline reaches 36.8% reference-LF success at 2.4 pp loss. Its rate 3×10⁻⁵ and coefficient 0.00528 are selected jointly on development LF/PCL. Candidate coefficients are {0.1, 1}/K̄, where K̄ averages the Gaussian mean-shift KL over six selected residual-GRPO checkpoints on common nominal states.')
p('Ideal routing sends nominal episodes to the frozen planner and selects fault-specific models without the retention limit. The residual contrast is 14.5 [2.2, 26.9] pp (Holm p = 0.048). Reevaluating original LF models with all robot–table collisions enabled and a ground plane added changes the procedure contrast by 0.2 [−0.9, 1.4] pp.')
t4=orig['TABLE IV. ORIGINAL NINE DIRECTIONAL TESTS']
table('TABLE IV. ORIGINAL NINE DIRECTIONAL TESTS',t4['headers'],t4['rows'],[.32,1.22,1.45,.49],
      'Effects and unadjusted two-sided 95% Welch intervals in pp; decisions use one-sided p-values with Holm correction. H2 contrasts LF and PCL effects. H4 reuses single-shift models under combined shifts: settings show gain over frozen before → after, and effects measure the reduction. H5 uses ideal routing.')

h('V. COMPACT OUTPUT CORRECTIONS')
sub('A. Recovery with a constant correction')
p('A state-independent correction recovers reference-LF performance. Averaging the trained residual GRPO head over nominal training-object states and using this constant output gives 66.7% success, compared with 64.2% for the source network. No further training is required. Table V includes run-level intervals and directly trained constant controls; the frozen planner still supplies state-dependent grasp proposals.')
table('TABLE V. REFERENCE-LF CORRECTION DIAGNOSTICS', ['Correction','Success\n[95% CI]','Nominal loss\n[95% CI]'],
      [[r['name']+(' ('+r['parameters']+')' if r['parameters'] in ('24','48') else ''),statfmt(r['success']),statfmt(r['loss'])] for r in S['constant_diagnostics']],
      [1.14,1.17,1.17],
      'n = 3 per row; success in %, loss in pp. Network and post-training mean share source models; their paired success difference is 2.4 [−6.2, 11.1] pp. Direct constant + scale selects on LF; the 24-parameter arms select jointly on LF/PCL with fixed sampling scale.')
p('This simplification is less effective elsewhere. At the retrained 75% posture, network-to-mean replacement changes success from 41.1% to 36.7%. Under two-finger lock, the network, its post-training mean and a directly trained constant reach 36.5%, 32.5% and 26.8%. A compact baseline is therefore informative alongside a correction network.')

sub('B. Coordinates and correction allocation')
p('For a frozen linear mean head m₀(s) = Wh₀(s) + b, let d = σ₀ ⊙ c. Then a scaled constant correction has the exact output-bias representation')
equation([('sub','m','0'),'(s) + d = W',('sub','h','0'),'(s) + (b + d).'],3)
p('Raw and scaled coordinates represent the same constant shifts. Their selected configurations reach 27.1% and 67.6% success, respectively, but a larger raw-coordinate rate also reaches 68.7% at 8.5 pp loss. Coordinates change the optimization path and the operating point selected by the retention rule. With Adam and negligible numerical stabilizer, reparameterizing d = σ₀ ⊙ c approximately scales effective output-coordinate step sizes by σ₀. This explains why coordinate choice is a direct control to examine, alongside network capacity.')
p('Checkpoint diagnostics locate the resulting movement. Residual GRPO allocates 87% of absolute mean movement to the 12 largest-scale dimensions, versus 62% for full-weight GRPO; 25% of its squared movement is state variation. After tanh and joint limits, mean RFJ3 target change is 1.29 rad versus 0.28 rad. At retrained 75%, residual GRPO instead has a 0.46 cosine with its reference correction and a 0.36 rad RFJ3 edit. These measurements connect the posture boundary to a changed learned correction.')

h('VI. CONCLUSION')
p('Residual procedures provide stronger reference little-finger recovery across overlapping observed nominal costs, and a substantial selected-configuration difference remains after full-weight grid refinement. At 75% lock posture the same-server contrast is 0.4 [−2.7, 3.5] pp. A constant output correction reproduces strong reference-lock recovery with a compact representation. The study makes recovery, retention cost and fault-posture coverage explicit parts of choosing an update for this pretrained planner.')
h('ACKNOWLEDGMENT')
p('Claude (Anthropic) and ChatGPT/Codex (OpenAI) assisted with drafting, editing, analysis and plotting code. The authors reviewed the text, code and results.','figure caption')
p('Data and code: https://github.com/woxihuanni99/Adapting-Pretrained-Dexterous-Grasping-Policies-to-Deployment-Shifts','figure caption')
h('REFERENCES')
for i,key in enumerate(citations,1):
    if i==25:
        b=D.add_paragraph();b.paragraph_format.space_before=Pt(0);b.paragraph_format.space_after=Pt(0)
        b.paragraph_format.line_spacing=Pt(1);b.paragraph_format.keep_with_next=True
        r=b.add_run();r.font.size=Pt(1);r.add_break(WD_BREAK.COLUMN)
    p(f'[{i}] '+BIB[key],'references')
