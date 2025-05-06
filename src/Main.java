public class Main {
    public static void main(String[] args) {
        LogisticRegression sm = new LogisticRegression();
        double[] y_true = {1, 0, 0, 0, 0};
        double[] y_pred = {10, 5, 2, 1, 4};

        double crossEntropyLoss = sm.crossEntropyLoss(y_pred, y_true);
        System.out.println();
    }
}
