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
        // Eğitimin başarıyla tamamlandığını doğrulamak için uygun assertion ekleyin
        assertNotNull(trainingData);
    }

    @Test
    public void testEvaluateModel() {
        DataModel testData = new DataModel(0.5, 0.5, 1.0);
        aiService.evaluateModel(testData);
        // Modelin değerlendirilmesinin doğru çalıştığını kontrol edin
        assertNotNull(testData); // Bu, gerçek test kodu ile değiştirilmeli
    }
}
