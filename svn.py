
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
second=time.time()

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
historySVM = SVM.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
modelEvaluation(predictions_SVM, Test_Y)
