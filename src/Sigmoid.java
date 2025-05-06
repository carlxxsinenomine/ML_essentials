public class Sigmoid {
    public double linearScore;

    public double sigmoid(float linearScore) {
        return 1 / (1.0 + Math.exp(-linearScore));
    }
}
