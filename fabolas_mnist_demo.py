
from HPOFramework.HPOBaseClasses import Hyperpipe, PipelineElement
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

mnist = fetch_mldata('MNIST original')
train_data, test_data, train_labels, test_labels = train_test_split(
    mnist.data/255.,
    mnist.target,
    test_size=1/7
)

cv = KFold(n_splits=3, shuffle=True, random_state=0)

n_train_data = len(train_data)
pipe = Hyperpipe(
    'god',
    cv,
    optimizer='fabolas',
    optimizer_params={
        'n_init': 7, 'n_min_train_data': int(n_train_data/9000), 'n_train_data': n_train_data,
        'num_iterations': 9, 'subsets': [2048, 4096, 8192]
    })
pipe.add(PipelineElement.create('svc', {'C': [-10, 10, 1.0], 'gamma': [-10, 10, 0.0]}))

pipe.fit(train_data, train_labels)
predicted_labels = pipe.predict(test_data)
print(classification_report(predicted_labels, test_labels))
