package com.example.ai.service;

import com.example.ai.model.DataModel;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;

public class AIService {
    private final MultiLayerNetwork model;

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
        // Eğitim verilerini hazırlama
        int dataSize = dataList.size();
        INDArray input = Nd4j.create(dataSize, 2); // 2 boyutlu matris
        INDArray labels = Nd4j.create(dataSize, 1); // 2 boyutlu matris

        for (int i = 0; i < dataSize; i++) {
            DataModel data = dataList.get(i);
            input.putRow(i, Nd4j.create(new double[]{data.getFeature1(), data.getFeature2()}));
            labels.putRow(i, Nd4j.create(new double[]{data.getLabel()}));
        }

        DataSet dataSet = new DataSet(input, labels);
        DataSetIterator iterator = new ListDataSetIterator<>(dataSet.asList(), 10);

        // Modeli eğitme
        int nEpochs = 1000;
        for (int i = 0; i < nEpochs; i++) {
            model.fit(iterator);
            if (i % 100 == 0) {
                System.out.println("Score at iteration " + i + " is " + model.score());
            }
        }
    }

    public void evaluateModel(DataModel testData) {
        INDArray input = Nd4j.create(new double[][]{{testData.getFeature1(), testData.getFeature2()}}); // 2 boyutlu matris
        INDArray output = model.output(input);
        System.out.println("Model Output: " + output);
    }
}
