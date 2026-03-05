# Fairlogue
Fairlogue, a toolkit for intersectional fairness analysis in clinical machine learning models. This Python-based modular toolkit consists of 3 components that will allow you to quantify and contextualize intersectional biases. 
The first component of the toolkit computes common fairness metrics (Equalized Odds, Demographic Parity, and Equal Opportunity Gap) along both single-axis demographics and intersectional demographics to provide a baseline
understanding of the disparities present in the model. 

The second component is a faithful translation of prior work<sup>1</sup> that allows for quantifying intersectional biases under group membership based counterfactual scenarios for models with an explicit treatment variable.

The third component is a generalized and extended version of component 2 that allows for quantifying intersectional biases under the same counterfactual scenarios but is applicable to a wider range of potential model types including models without a treatment variable.






Ref:
1. Wastvedt, S., Huling, J., & Wolfson, J. (2023). An intersectional framework for counterfactual fairness in risk prediction. Biostatistics, kxad021. https://doi.org/10.1093/biostatistics/kxad021  
