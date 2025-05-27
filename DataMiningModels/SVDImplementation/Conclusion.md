# 🧠 PCA vs. Full Feature Model: Analysis

## 🎯 Accuracy
Both models — the one trained on all 13 features (**full feature model**) and the one trained after PCA reduction to 2 components — showed **perfect or near-perfect accuracy**.  
This indicates that the two principal components retained most of the information necessary for classification.

## 📊 Visualization
Only the **PCA-based model** could be visualized using a 2D scatter plot.  
The decision boundaries between classes are clearly shown, making the classifier’s logic interpretable at a glance.  
The **full feature model**, operating in 13D, is not visually interpretable — we rely solely on numerical results.

## ✅ Conclusion: Was Dimensionality Reduction Worthwhile?
**Yes** — PCA significantly reduced dimensionality while preserving performance.

### Benefits:
- **Interpretability:** We gain visual insight into the data.
- **Efficiency:** Training and inference are faster on smaller input dimensions.
- **Generalization:** Reducing complexity can reduce overfitting on small datasets.

---

🔁 **Therefore**, PCA was a worthwhile step — offering similar accuracy with clearer insight and lower complexity.

