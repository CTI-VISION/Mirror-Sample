# Mirror-Sample

This is the code repository for NeurIPS 2021 paper  "Yin Zhao, Minquan Wang, Longjun Cai, Reducing the Covariate Shift by Mirror Samples in Cross Domain Alignment"
The code will come soon after some disclosure procedure of our affiliation.


### Reducing the Covariate Shift by Mirror Samples in Cross Domain Alignment

#### Yin Zhao, Minquan Wang, Longjun Cai

### abstract 
Eliminating the covariate shift cross domains is one of the common methods to deal with the issue of domain shift in visual unsupervised domain adaptation.
However, current alignment methods, especially the prototype based or sample-level based methods neglect the structural properties of the underlying distribution and even break the condition 
of covariate shift.
To relieve the limitations and conflicts, we introduce a novel concept named (virtual) mirror, which represents the equivalent sample in another domain.
The equivalent sample pairs, named mirror pairs reflect the natural correspondence of the empirical distributions.
Then a mirror loss, which aligns the mirror pairs cross domains, is constructed to enhance the alignment of the domains.
The proposed method does not distort the internal structure of the underlying distribution.
We also provide theoretical proof that the mirror samples and mirror loss have better asymptotic properties in reducing the domain shift.
By applying the virtual mirror and mirror loss to the generic unsupervised domain adaptation model, we achieved consistent superior performance on several mainstream benchmarks.
