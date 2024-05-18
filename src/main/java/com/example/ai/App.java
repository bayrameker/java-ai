package com.example.ai;

import com.example.ai.model.DataModel;
import com.example.ai.service.AIService;

import java.util.ArrayList;
import java.util.List;

public class App {
    public static void main(String[] args) {
        AIService aiService = new AIService();

        // Eğitim verilerini oluşturma
        List<DataModel> trainingData = new ArrayList<>();
        trainingData.add(new DataModel(0.1, 0.2, 0.3));
        trainingData.add(new DataModel(0.2, 0.3, 0.5));
        trainingData.add(new DataModel(0.3, 0.4, 0.7));
        trainingData.add(new DataModel(0.4, 0.5, 0.9));

        aiService.trainModel(trainingData);

        // Değerlendirme verisi ile modeli test etme
        DataModel testData = new DataModel(0.5, 0.5, 1.0);
        aiService.evaluateModel(testData);
    }
}
