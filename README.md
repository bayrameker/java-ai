

# AI Project

This project is a simple AI application built with Java, using the Deeplearning4j library for deep learning, ND4J for numerical computations, and DataVec for data preprocessing. The project demonstrates basic machine learning model training and evaluation.

## Table of Contents

- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [Testing the Project](#testing-the-project)
- [Code Explanation](#code-explanation)
- [Improvements](#improvements)

## Project Structure

The project follows a structured file layout:

```
ai-project
├── src
│   ├── main
│   │   ├── java
│   │   │   └── com
│   │   │       └── example
│   │   │           └── ai
│   │   │               ├── App.java
│   │   │               ├── model
│   │   │               │   └── DataModel.java
│   │   │               └── service
│   │   │                   └── AIService.java
│   │   └── resources
│   └── test
│       ├── java
│       │   └── com
│       │       └── example
│       │           └── ai
│       │               └── service
│       │                   └── AIServiceTest.java
```

## Dependencies

The project uses Maven for dependency management. The primary dependencies include:

- Deeplearning4j
- ND4J
- DataVec
- JUnit
- SLF4J

These dependencies are specified in the `pom.xml` file.

## Setup

### Prerequisites

- Java 8 or later
- Maven 3.6 or later

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/ai-project.git
    cd ai-project
    ```

2. Install dependencies and compile the project:

    ```sh
    mvn clean compile
    ```

## Running the Project

To run the project, use the following Maven command:

```sh
mvn exec:java -Dexec.mainClass="com.example.ai.App"
```

This command executes the main application class `App.java`.

## Testing the Project

To run the tests, use the following Maven command:

```sh
mvn test
```

This command runs all the tests defined in the `src/test` directory.

## Code Explanation

### App.java

The main application entry point. It sets up the training data, trains the model, and evaluates it with test data.

```java
package com.example.ai;

import com.example.ai.model.DataModel;
import com.example.ai.service.AIService;

import java.util.ArrayList;
import java.util.List;

public class App {
    public static void main(String[] args) {
        AIService aiService = new AIService();

        // Creating training data
        List<DataModel> trainingData = new ArrayList<>();
        trainingData.add(new DataModel(0.1, 0.2, 0.3));
        trainingData.add(new DataModel(0.2, 0.3, 0.5));
        trainingData.add(new DataModel(0.3, 0.4, 0.7));
        trainingData.add(new DataModel(0.4, 0.5, 0.9));

        aiService.trainModel(trainingData);

        // Evaluating the model with test data
        DataModel testData = new DataModel(0.5, 0.5, 1.0);
        aiService.evaluateModel(testData);
    }
}
```

### DataModel.java

Defines the data structure for the input features and the label.

```java
package com.example.ai.model;

public class DataModel {
    private double feature1;
    private double feature2;
    private double label;

    public DataModel(double feature1, double feature2, double label) {
        this.feature1 = feature1;
        this.feature2 = feature2;
        this.label = label;
    }

    // Getters and setters
}
```

### AIService.java

Handles the creation, training, and evaluation of the machine learning model.

```java
package com.example.ai.service;

import com.example.ai.model.DataModel;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.ListDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

public class AIService {
    private MultiLayerNetwork model;

    public AIService() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Sgd(0.1))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(2)
                        .nOut(3)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(3)
                        .nOut(1)
                        .activation(Activation.IDENTITY)
                        .build())
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));
    }

    public void trainModel(List<DataModel> dataList) {
        // Preparing training data
        int dataSize = dataList.size();
        INDArray input = Nd4j.create(dataSize, 2); // 2D matrix
        INDArray labels = Nd4j.create(dataSize, 1); // 2D matrix

        for (int i = 0; i < dataSize; i++) {
            DataModel data = dataList.get(i);
            input.putRow(i, Nd4j.create(new double[]{data.getFeature1(), data.getFeature2()}));
            labels.putRow(i, Nd4j.create(new double[]{data.getLabel()}));
        }

        DataSet dataSet = new DataSet(input, labels);
        DataSetIterator iterator = new ListDataSetIterator<>(dataSet.asList(), 10);

        // Training the model
        int nEpochs = 1000;
        for (int i = 0; i < nEpochs; i++) {
            model.fit(iterator);
            if (i % 100 == 0) {
                System.out.println("Score at iteration " + i + " is " + model.score());
            }
        }
    }

    public void evaluateModel(DataModel testData) {
        INDArray input = Nd4j.create(new double[][]{{testData.getFeature1(), testData.getFeature2()}}); // 2D matrix
        INDArray output = model.output(input);
        System.out.println("Model Output: " + output);
    }
}
```

### AIServiceTest.java

Contains unit tests for the AIService class.

```java
package com.example.ai.service;

import com.example.ai.model.DataModel;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

public class AIServiceTest {
    private AIService aiService;

    @Before
    public void setUp() {
        aiService = new AIService();
    }

    @Test
    public void testTrainModel() {
        List<DataModel> trainingData = new ArrayList<>();
        trainingData.add(new DataModel(0.1, 0.2, 0.3));
        trainingData.add(new DataModel(0.2, 0.3, 0.5));
        trainingData.add(new DataModel(0.3, 0.4, 0.7));
        trainingData.add(new DataModel(0.4, 0.5, 0.9));

        aiService.trainModel(trainingData);
        // Verify that training data is not null
        assertNotNull(trainingData);
    }

    @Test
    public void testEvaluateModel() {
        DataModel testData = new DataModel(0.5, 0.5, 1.0);
        aiService.evaluateModel(testData);
        // Verify that test data is not null
        assertNotNull(testData);
    }
}
```

## Improvements

- **Data Normalization**: Normalize the input data to improve model performance.
- **Cross-Validation**: Implement cross-validation to better assess model performance.
- **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and model architectures.
- **Logging**: Implement more detailed logging for better monitoring of the training process.

