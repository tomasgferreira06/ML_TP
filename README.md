# ML_TP

## tratamento outliers
“Apesar de existirem valores considerados outliers segundo o critério IQR, estes representam medições quimicamente plausíveis e características reais do vinho (por exemplo, teores elevados de açúcar ou dióxido de enxofre).

Assim, optou-se por não remover outliers, garantindo que a variabilidade natural do processo de produção é mantida e que os modelos aprendem a lidar com amostras extremas.”



## Alpha nas redes neuronais


### Cenário A: Alpha BAIXO (0.0001) - OVERFITTING
    # Modelo com alpha baixo
    nn = MLPClassifier(alpha=0.0001, ...)
    nn.fit(X_train, y_train)
    
    # O que acontece:
    # - Vê no treino: "Vinho com álcool=13.2 e sulphates=0.68 é qualidade 7"
    # - Decora: "Se álcool=13.2 E sulphates=0.68 → sempre qualidade 7"
    # - No teste: vê álcool=13.1 e sulphates=0.69 → ERRA!
    # - Decorou os valores exatos, não aprendeu o padrão geral `


### Cenário B: Alpha MUITO ALTO (0.5) - UNDERFITTING

    # Modelo com alpha muito alto
    nn = MLPClassifier(alpha=0.5, ...)
    nn.fit(X_train, y_train)
    
    # O que acontece:
    # - Pesos ficam TÃO pequenos que o modelo não consegue aprender padrões
    # - É como estudar SÓ títulos sem ler o conteúdo



### Notes:

    -> hyperparameter tuning (by hand, etc...)

    -> refined hyperparameters

    -> alternativa a grid search, optuna

    30 NN:
    -> with the same hyperparameters, start a neural network with different weights, learning rate.
    -> fazer a média, apresentar a desvia padrão.     


## Melhores Configs NN

    📊 TOP 5 MELHORES CONFIGURAÇÕES (por Test Accuracy):
    ------------------------------------------------------------------------------------------

    2º lugar:
    Parâmetros: {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (128,), 'learning_rate': 'constant', 'learning_rate_init': 0.01, 'max_iter': 2500, 'solver': 'sgd'}
    Test Acc:  0.7716 ± 0.0037
    Train Acc: 0.8132 ± 0.0194
    Gap:       0.0416 ± 0.0205

    1º lugar:
    Parâmetros: {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (64,), 'learning_rate': 'adaptive', 'learning_rate_init': 0.01, 'max_iter': 2500, 'solver': 'sgd'}
    Test Acc:  0.7716 ± 0.0060
    Train Acc: 0.7998 ± 0.0146
    Gap:       0.0282 ± 0.0102

    3º lugar:
    Parâmetros: {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': (128,), 'learning_rate': 'constant', 'learning_rate_init': 0.01, 'max_iter': 2500, 'solver': 'sgd'}
    Test Acc:  0.7706 ± 0.0096
    Train Acc: 0.8083 ± 0.0161
    Gap:       0.0377 ± 0.0221

    4º lugar:
    Parâmetros: {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (64,), 'learning_rate': 'constant', 'learning_rate_init': 0.01, 'max_iter': 2500, 'solver': 'sgd'}
    Test Acc:  0.7706 ± 0.0096
    Train Acc: 0.7936 ± 0.0171
    Gap:       0.0230 ± 0.0097

    5º lugar:
    Parâmetros: {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (128,), 'learning_rate': 'adaptive', 'learning_rate_init': 0.01, 'max_iter': 2500, 'solver': 'sgd'}
    Test Acc:  0.7696 ± 0.0037
    Train Acc: 0.7956 ± 0.0085
    Gap:       0.0259 ± 0.0076