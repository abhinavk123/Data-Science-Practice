#SVM 
predictions_SVM_binarise = label_binarize(predictions_SVM, classes=[0, 1, 2])
fpr_SVM = dict()
tpr_SVM = dict()
roc_auc_SVM = dict()
for i in range(3):
    fpr_SVM[i], tpr_SVM[i], _ = roc_curve(Test_Y_binarise[:, i], predictions_SVM_binarise[:, i])
    roc_auc_SVM[i] = auc(fpr_SVM[i], tpr_SVM[i])

# Compute micro-average ROC curve and ROC area
fpr_SVM["micro"], tpr_SVM["micro"], _ = roc_curve(Test_Y_binarise.ravel(), predictions_SVM_binarise.ravel())
roc_auc_SVM["micro"] = auc(fpr_SVM["micro"], tpr_SVM["micro"])

# First aggregate all false positive rates
all_fpr_SVM = np.unique(np.concatenate([fpr_SVM[i] for i in range(3)]))

# Then interpolate all ROC curves at this points
mean_tpr_SVM = np.zeros_like(all_fpr_SVM)
for i in range(3):
    mean_tpr_SVM += interp(all_fpr_SVM, fpr_SVM[i], tpr_SVM[i])

# Finally average it and compute AUC
mean_tpr_SVM /= 3

# Plot all ROC curves
plt.figure()
plt.plot(fpr_SVM["micro"], tpr_SVM["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc_SVM["micro"]),
         color='deeppink', linestyle=':', linewidth=5)


plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
