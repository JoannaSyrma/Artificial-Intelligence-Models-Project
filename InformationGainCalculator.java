import java.util.*;

public class InformationGainCalculator {

    public double calculateIG(List<Integer> classesVector, List<Integer> feature) {//feature = 0/1 classesVector=
        HashSet<Integer> classes = new HashSet<>(classesVector);

        double HC = 0;
        for (Integer c : classes) {
            double PC = (double) countOccurrences(classesVector, c) / classesVector.size(); //πιθανότητα εμφάνισης μιας συγκεκριμένης κλάσης (class) στο dataset
            HC += -PC * log2(PC); //εντροπία της μεταβλητής κλάσης (class variable) στο dataset
        }

        HashSet<Integer> featureValues = new HashSet<>(feature);
        double HCFeature = 0;
        for (Integer value : featureValues) {
            double pf = (double) countOccurrences(feature, value) / feature.size(); //count occurences of value
            List<Integer> indices = getIndices(feature, value);

            List<Integer> classesOfFeat = getClassesAtIndex(classesVector, indices);
            for (Integer c : classes) {
                double pcf = (double) countOccurrences(classesOfFeat, c) / classesOfFeat.size();
                if (pcf != 0) {
                    double tempH = -pf * pcf * log2(pcf);
                    HCFeature += tempH;
                }
            }
        }

        double ig = HC - HCFeature;
        return ig;
    }

    private int countOccurrences(List<Integer> list, int value) {
        int count = 0;
        for (Integer element : list) {
            if (element == value) {
                count++;
            }
        }
        return count;
    }

    private List<Integer> getIndices(List<Integer> list, int value) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) == value) {
                indices.add(i);
            }
        }
        return indices;
    }

    private List<Integer> getClassesAtIndex(List<Integer> classesVector, List<Integer> indices) {
        List<Integer> classesOfFeat = new ArrayList<>();
        for (Integer index : indices) {
            classesOfFeat.add(classesVector.get(index));
        }
        return classesOfFeat;
    }

    private double log2(double x) {
        return Math.log(x) / Math.log(2);
    }

    public static void main(String[] args) {
        // Example usage
        InformationGainCalculator calculator = new InformationGainCalculator();
        List<Integer> classesVector = List.of(1, 1, 0, 1, 0, 0, 1, 0);
        List<Integer> feature = List.of(1, 0, 1, 0, 1, 0, 1, 0);
        double ig = calculator.calculateIG(classesVector, feature);
        System.out.println("Information Gain: " + ig);
    }
}
