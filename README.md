# DEML

## Abstract
(1) Background: Synergistic drug combinations have demonstrated effective therapeutic effects in cancer treatment. Deep learning methods accelerate identification of novel drug combinations by reducing the search space. However, potential adverse drug–drug interactions (DDIs), which may increase the risks for combination therapy, cannot be detected by existing computational synergy prediction methods. (2) Methods: We propose DEML, an ensemble-based multi-task neural network, for the simultaneous optimization of five synergy regression prediction tasks, synergy classification and DDI classification task. DEML uses chemical and transcriptomics information as input. DEML adapts the novel hybrid ensemble layer structure to construct higher order repre-sentation by different perspective. The task-specific fusion layer of DEML joint representations for each task by gating mechanism. (3) Results: For Loewe synergy prediction task, DEML overper-forms the state-of-the-art synergy prediction method with an improvement of 14.5% and 6.4% for mean squared error and Pearson correlation coefficient. Owing to soft parameters sharing and ensemble learning, DEML alleviates multi-task learning ‘seesaw effect’ problem and shows no performance loss on other tasks. (4) Conclusions: DEML has superior ability for prediction of drug pairs with high confident and less adverse DDIs, and provide a promising way to guideline novel combination therapy strategies for cancer treatment. 

## Requirement
torch v1.7.0

numpy v1.21.2

pandas  v1.3.4

scipy v1.7.3

scikit-learn v0.23.2

## Data

drugfeature.csv refers to the drug chemical features.

cell-line-feature_express_extract.csv refers to the cell lines gene expression features.

drugdrug_extract28newddi.csv refers to the data label.

The data resource can find in the paper.
