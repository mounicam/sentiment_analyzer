# Changelog

## v0.3.1

## New features
- Implement SquadQA tensorizer in TorchScript (#1211)
- Add session data source for df (#1202)
- Dynamic Batch Scheduler Implementation (#1200)
- Implement loss aware sparsifier (#1204)
- Ability to Fine-tune XLM-R for NER on CoNLL Datasets (#1201)
- TorchScriptify Tokenizer after training (#1191)
- Linear Layer only blockwise sparsifier (#478)
- Adding performance graph to pytext models (#1192)
- Enable inference on GPUs by moving tensors to specified device (#472)
- Add support for learning from soft labels for Squad (MRC) models (#1188)
- Create byte-aware model that can make byte predictions (#468)
- Minimum Trust Lamb (#1186)
- Allow model to take byte-level input and make byte-level prediction (#1187)
- Scheduler with Warmup (#1184)
- Implement LAMB optimizer (#1183)
- CyclicLRScheduler (#1157)
- PyText Entity Linking: ELTask and ELMetricReporter (#1165)

## Bug fixes
- Don't upgrade if Tensorizer already given (#504)
- avoid torchscriptify on a ScriptModule (#1214)
- Make tensorboard robust to NaN and Inf in model params (#1206)
- Fix circleCLI Test broken in D19027834 (#1205)
- Fix small bug in pytext vocabulary (#401)
- Fix CircleCI failure caused by black and regex (#1199)
- Fix CircleCI (#1194)
- Fix Circle CI Test broken by D18880705 (#1190)
- fix weight load for new fairseq checkpoints (#1189)
- Fix Heirarchical intent and slot filling demo is broken (#1012) (#1151)
- Fix index error in dict embedding when exported to Caffe2 (#1182)
- Fix zero loss tensor in SquadOutputLayer (#1181)
- qa fix for ignore_impossible=False

## Other

- Printing out error's underlying reason (#1227)
- tidy file path in help text for invocation of docnn.json example (#1221)
- PyText option to disable CUDA when testing. (#1223)
- make augmented lstm compatible w other lstms (#1224)
- Vocab recursive lookup (#1222)
- Fix simple typo: valus -> value (#1219)
- support using RoundRobin ProcessGroup in Distributed training (#1213)
- Use PathManager for all I/O (#1198)
- Make PathManager robust to API changes in fvcore (#1196)
- Support for TVM training (BERT) (#1210)
- Exit LM task if special token exists in text for ByteTensorizer (#1207)
- Config adapter for pytext XLM (#1172)
- Use TensorizerImpl for both training and inference for BERT, RoBERTa and XLM tensorizer (#1195)
- Replace gluster paths with local file paths for NLG configs (#1197)
- Make BERT Classification compatible with TSEs that return Encoded Layers.
- implement BertTensorizerImpl and XLMTensorizerImpl (#1193)
- Make is_input field of tensorizer configurable (#474)
- BERTTensorizerBaseImpl to reimplement BERTTensorizerBase to be TorchScriptable (#1163)
- Improve LogitsWorkflow to handle dumping of raw inputs and multiple output tensors (#683)
- Accumulative blockwise pruning (#1170)
- Patch for UnicodeDecodeError due to BPE. (#1179)
- Add pre-loaded task as parameter to caffe2 batch prediction API
- Specify CFLAGS to install fairseq in MacOS (#1175)
- Resolve dependency conflict by specifying python-dateutil==2.8.0 (#1176)
- Proper training behavior if setting do_eval=False (#1155)
- Make DeepCNNRepresentation torchscriptable (#453)


## v0.3.0

## New Features
**RoBERTa and XLM-R**
- Integrate XLM-R into PyText (#1120)
- Consolidate BERT, XLM and RobERTa Tensorizers (#1119)
- Add XLM-R for joint model (#1135)
- Open source Roberta (#1032)
- Simple Transformer module components for RoBERTa (#1043)
- RoBERTa models for document classification (#933)
- Enable MLM training for RobertaEncoder (#1126)
- Standardize RoBERTa Tensorizer Vocab Creation (#1113)
- Make RoBERTa usable in more tasks including QA (#1017)
- RoBERTa-QA JIT (#1088)
- Unify GPT2BPE Tokenizer (#1110)
- Adding Google SentencePiece as a Tokenizer (#1106)

**TorchScript support**
- General torchscript module (#1134)
- Support torchscriptify XLM-R (#1138)
- Add support for torchscriptification of XLM intent slot models (#1167)
- Script xlm tensorizer (#1118)
- Refactor ScriptTensorizer with general tensorize API (#1117)
- ScriptXLMTensorizer (#1123)
- Add support for Torchscript export of IntentSlotOutputLayer and CRF (#1146)
- Refactor ScriptTensorizor to support both text and tokens input (#1096)
- Add torchscriptify API in tokenizer and tensorizer (#1055)
- Add more stats in torchscript latency script (#1044)
- Exported Roberta torchscript model include both traced_model and pre-processing logic (#1013)
- Native Torchscript Wordpiece Tokenizer Op for BERTSquadQA, Torchscriptify BertSQUADQAModel (#879)
- TorchScript-ify BERT training (#887)
- Modify Return Signature of TorchScript BERT (#1058)
- Implement BertTensorizer and RoBERTaTensorizer in TorchScript (#1053)

**Others**
- FairseqModelEnsemble class (#1116)
- Inverse Sqrt Scheduler (#1150)
- Lazy modules (#1039)
- Adopt Fairseq MemoryEfficientFP16Optimizer in PyText (#910)
- Add RAdam (#952)
- Add AdamW (#945)
- Unify FP16&FP32 API (#1006)
- Add precision at recall metric (#1079)
- Added PandasDataSource (#1098)
- Support testing Caffe2 model (#1097)
- Add contextual feature support to export for Seq2Seq models
- Convert matmuls to quantizable nn.Linear modules (#1304)
- PyTorch eager mode implementation (#1072)
- Implement Blockwise Sparsification (#1050)
- Support Fairseq FP16Optimizer (#1008)
- Make FP16OptimizerApex wrapper on Apex/amp (#1007)
- Remove vocab from cuda (#955)
- Add dense input to XLMModel (#997)
- Replace tensorboardX with torch.utils.tensorboard (#1003)
- Add mentioning of mixed precision training support (#643)
- Sparsification for CRF transition matrix (#982)
- Add dense feature normalization to Char-LSTM TorchScript model. (#986)
- Cosine similarity support for BERT pairwise model training (#967)
- Combine training data from multiple sources (#953)
- Support visualization of word embeddings in Tensorboard (#969)
- Decouple decoder and output layer creation in BasePairwiseModel (#973)
- Drop rows with insufficient columns in TSV data source (#954)
- Add use_config_from_snapshot option(load config from snapshot or current task) (#970)
- Add predict function for NewTask (#936)
- Use `create_module` to create CharacterEmbedding (#920)
- Add XLM based joint model
- Add `ConsistentXLMModel` (#913)
- Optimize Gelu module for caffe2 export (#918)
- Save best model's sub-modules when enabled (#912)

## Documentation / Usability
- XLM-R tutorial in notebook (#1159)
- Update XLM-R OSS tutorial and add Google Colab link (#1168)
- Update "raw_text" to "text" in tutorial (#1010)
- Make tutorial more trivial (add git clone) (#1037)
- Changes to make tutorial code simpler (#1002)
- Fix datasource tutorial example (#998)
- Handle long documents in squad qa datasource and models (#975)
- Fix pytext tutorial syntax (#971)
- Use torch.equal() instead of "==" in Custom Tensorizer tutorial (#939)
- Remove and mock doc dependencies because readthedocs is OOM (#983)
- Fix Circle CI build_docs error (#959)
- Add OSS integration tests: DocNN (#1021)
- Print model into the output log (#1127)
- Migrate pytext/utils/torch.py logic into pytext/torchscript/ for long term maintainability (#1082)
- Demo datasource fix + cleanup (#994)
- Documentation on the config files and config-related commands (#984)
- Config adapter old data handler helper (#943)
- Nicer gen_config_impl (#944)

## Deprecated Features
- Remove DocModel_Deprecated (#916)
- Remove RNNGParser_Deprecated, SemanticParsingTask_Deprecated, SemanticParsingCppTask_Deprecate, RnngJitTask,
- Remove QueryDocumentTask_Deprecated(#926)
- Remove LMTask_Deprecated and LMLSTM_Deprecated (#882)
- CompositionDataHandler to fb/deprecated (#963)
- Delete deprecated Word Tagging tasks, models and data handlers (#910)

## Bug Fixes
- Fix caffe2 predict (#1103)
- Fix bug when tensorizer is not defined (#1169)
- Fix multitask metric reporter for lr logging (#1164)
- Fix broken gradients logging and add lr logging to tensorboard (#1158)
- Minor fix in blockwise sparsifier (#1130)
- Fix clip_grad_norm API (#1143)
- Fix for roberta squad tensorizer (#1137)
- Fix multilabel metric reporter (#1115)
- Fixed prepare_input in tensorizer (#1102)
- Fix unk bug in exported model (#1076)
- Fp16 fixes for byte-lstm and distillation (#1059)
- Fix clip_grad_norm_ if grad_norm > max_norm > 0: TypeError: '>' not supported between instances of 'float' and 'NoneType' (#1054)
- Fix context in multitask (#1040)
- Fix regression in ensemble trainer caused by recent fp16 change (#1033)
- ReadTheDocs OOM fix with CPU Torch (#1027)
- Dimension mismatch after setting max sequence length (#1154)
- Allow null learning rate (#1156)
- Don't fail on 0 input (#1104)
- Remove side effect during pickling PickleableGPT2BPEEncoder
- Set onnx==1.5.0 to fix CircleCI build temporarily (#1014)
- Complete training loop gracefully even if no timing is reported (#1128)
- Propagate min_freq for vocab correctly (#907)
- Fix gen-default-config with Model param (#917)
- Fix torchscript export for PyText modules (#1125)
- Fix label_weights in DocModel (#1081)
- Fix label_weights in bert models (#1100)
- Fix config issues with Python 3.7 (#1066)
- Temporary fix for Fairseq dependency (#1026)
- Fix MultipleData by making tensorizers able to initialize from multiple data sources (#972)
- Fix bug in copy_unk (#964)
- Division by Zero bug in MLM Metric Reporter (#968)


## v0.2.2

*Note:* this is the last release with _Deprecated classes. Those classes will be removed in the next release.

**New Features:**
- DeepCNN Representation for word tagging
- Combine KLDivergenceBCELoss with SoftHardBCELoss and F.cross_entropy() in CrossEntropyLoss (#689)
- add dense feature support for doc model (#710)
- add torchscript quantizaiton support in pytext
- pytext multi-label support (#731)
- open source transformer representations (#736)
- open source transformer based models - data, tensorizers and tokenizer (#708)
- Create AlternatingRandomizedBatchSampler (#737)
- open source MaskedLM and BERT models (#734)
- Support bytes input in word tagging model OSS (#745)
- open source extractive question answering models (#742)
- torchscriptify for ensemle task
- enabled lmlstm labels exporting (#767)
- Enable dense features in ByteTokensDocumentModel (#763)
- created bilstm dropout condition (#769)
- enabled lmlstm caffe2 exporting (#766)
- PolynomialDecayScheduler (#791)
- removed bilstm dependence on seq_lengths (#776)
- fp16 optimizer (#782)
- Add Dense Feature Normalization to FloatListTensorizer and DocModel (#859)
- Add Sparsifier component to PyText and L0-projection based sparsifier (#860)
- implemented cnn pooling for doc classification (#872)
- implemented bottleneck separable convolutions (#855)
- Add eps to Adam (#858)
- implemented mobile exporter (#785)
- support starting training from saved checkpoint (#824)
- implemented separable convolutions (#830)
- implemented gelu activations (#829)
- implemented causal convolutions (#811)
- implemented dilation for convolutions (#810)
- created weight norm option (#809)
- Ordered Neuron LSTM (#854)
- Add PersonalizedByteDocModel (#816)
- CNN based language models (#827)
- improve csv support in TSVDataSource (#777)
- Change default batch sampler DisjointMultitaskData to RoundRobinBatchSampler (#802)
- Support using serialized pretrained embedding file (#797)

**Documentation / Usability / Logging:**
- Fewer out-of-vocab print messages, with some stats (#697)
- Echo epoch number to console while training (#712)
- Separate timing for prediction and metric calculation. (#738)
- multi-label soft metrics (#754)
- changed lm metric reporting (#765)
- fix data source tutorial (#762)
- fix doc sphinx deprecation warning (#775)
- Add the ability to pass parameter values to gen-default-config (#856)
- Remove "pytext/" from paths in demo json config (#878)
- New documentation about hacking pytext and dealing with github. (#862)
- install_deps supports updates (#863)
- Reduce number of PEP print (#861)
- better error message for config with unknown component (#801)
- Add Recall at Precision Thresholds to Config (#792)
- implemented perplexity reductions for lm score reporting (#799)
- adapt prediction workflow to new design (#746)

**Bug fixes:**
- block sharded tsv eval/test fix (#698)
- Fix BoundaryPooling tracing (#713)
- fixes LMLSTM weight tying bug (#704)
- Fix duplicate entries in vocab (#721)
- Bugfix for trainer not reporting eval results (#740)
- Reintroduce metrics export in new task (#748)
- fix open source tests (#750)
- Fix missing init_tensorizers arg (#893)
- Fix intent slot metric reporter not working with byte offset (#883)
- Fix issue with some tensorizers still re-initializing vocab when loaded from saved state (#848)
- fixed overflow error in lm reporting (#831)
- fix BlockShardedTSVDataSource (#832)


## v0.2.1

(skipped because of packaging issues)


## v0.2.0

*Note:* This release makes the new data handler API the default and deprecates Task and Model classes using the old data handler API. We recommend that you migrate your models to the new API as soon as possible. More details here: http://...

**New Stuff**
- most tasks and models deprecated, replaced with better versions using the new data handler API
- performance improvements in metric reporter
- Add Multilingual TSV Data Source
- LabelSmoothedCrossEntropyLoss
- Support for pretrained word embedding in TokenTensorizer
- option to use pretrained embedding
- TorchScript export for document classification
- Improve log in trainer
- performance measurement: reporting tokens_per_second and updates_per_second
- Implement DocumentReader from DrQA in PyText (StackedBidirectionalRNN)
- improved and updated documentation
- Implement SWA(SGD|ADAM) and Adagrad Optimizers
- cache numerized data in memory
- TorchScript BPE tokenization
- CLI command to update configs
- Visualize gradients with tensorboard

*Many bug fixes and code clean-ups*


## v0.1.5

*Note:* this is a last release in 0.1.x. The next release will deprecate Task and Model base classes and make the improved API of the new data handler the default. You can start using it already by inheriting from NewTask. NewDocumentClassification and NewWordTaggingTask use this new API, and you can get the first example in the tutorial "Custom Data Format".

**New Stuff**
- add config adapter
  - PyText is very young and its API is still in flux, making the config files brittle
  - config files now have a version number reflecting the API at the time it was created
  - older versions can be loaded and internally transformed into newer versions
- better metrics and reporting
  - better training time tracking
  - cool new visualization of model state in TensorBoard
  - pretty results in the terminal
- improved distributed training
- torchscript export
- support for SQuAD dataset
- add AugmentedLSTM
- add dense features support
- new plugin system: command line option --include to import custom user classes (see tutorial "Custom Data Format" for example)

*Many bug fixes and code clean-ups*


## v0.1.4
**New Stuff**
- Refactor Metric Reporters to reduce coupling
- RNNG Improvements:
  - Support Pretrained embeddings in RNNG
  - Support GPU Training
  - More Test Coverage
  - Tensorboard Support
- Added `QueryDocumentPairwiseRankingModel`
- Distributed Training Improvments:
  - Sharded Data Loading to reduce memory consumption
  - Fix Several issues with race conditions and unserializable state
- Reduced GPU memory Consumption by skipping gradient computation on evaluation

*And lots of bug fixes*

**Known Issues**
PyText doesn't work with the new ONNX v1.4.0, so we have pinned it to 1.3.0 for now


## v0.1.3
 - Remove epoch_size param from DisjointMultitask, use target_task (or shortest) to set epoch_size

## v0.1.0

Initial version
