import java.util.HashMap;
import java.util.Map;

public class NaiveBayesClassifier {
    private Map<String, Integer> vocabulary;
    private Map<String, Integer> positiveClassCounts;
    private Map<String, Integer> negativeClassCounts;
    private int totalPositiveDocuments;
    private int totalNegativeDocuments;

    public NaiveBayesClassifier() {
        vocabulary = new HashMap<>();
        positiveClassCounts = new HashMap<>();
        negativeClassCounts = new HashMap<>();
        totalPositiveDocuments = 0;
        totalNegativeDocuments = 0;
    }

    public void train(String[] document, boolean isPositive) {
        if (isPositive) {
            totalPositiveDocuments++;
        } else {
            totalNegativeDocuments++;
        }

        Map<String, Integer> classCounts = isPositive ? positiveClassCounts : negativeClassCounts;

        for (String word : document) {
            vocabulary.putIfAbsent(word, 0);
            classCounts.put(word, classCounts.getOrDefault(word, 0) + 1);
            vocabulary.put(word, vocabulary.get(word) + 1);
        }
    }

    public String classify(String[] document) {
        double positiveProbability = calculateClassProbability(document, true);
        double negativeProbability = calculateClassProbability(document, false);

        return (positiveProbability > negativeProbability) ? "Positive" +positiveProbability : "Negative"
        +negativeProbability;
    }

    private double calculateClassProbability(String[] document, boolean isPositive) {
        Map<String, Integer> classCounts = isPositive ? positiveClassCounts : negativeClassCounts;
        int totalDocumentsInClass = isPositive ? totalPositiveDocuments : totalNegativeDocuments;

        double logProbability = 0.0;

        for (String word : document) {
            int wordCountInClass = classCounts.getOrDefault(word, 0);
            double probability = (double) (wordCountInClass + 1) / (totalDocumentsInClass + 2);
            logProbability += Math.log(probability);
        }

        return logProbability;
    }

    public static void main(String[] args) {
        NaiveBayesClassifier classifier = new NaiveBayesClassifier();

        // train tajinomiti me thetika paradeigmata
        classifier.train(new String[]{"good", "movie"}, true);
        classifier.train(new String[]{"excellent", "film"}, true);
        classifier.train(new String[]{"great", "movie"}, true);
        classifier.train(new String[]{"great", "movie"}, true);
        classifier.train(new String[]{"great", "movie"}, true);
        classifier.train(new String[]{"great", "movie"}, true);



        // train tajinomiti me arnitika paradeigmata
        classifier.train(new String[]{"bad", "movie"}, false);
        classifier.train(new String[]{"terrible", "film"}, false);
        classifier.train(new String[]{"great", "movie"}, false);



        //dokimi tajinomiti me agnosto egrafo 
        String[] testDocument = {"great", "film"};
        String result = classifier.classify(testDocument);

        System.out.println("Document is classified as: " + result);
    }
}
