/*
In a scenario such as this one, the computer has no idea what the relationship
between X and Y is. So it will make a guess. Say for example it guesses that Y = 10X +
10. It then needs to measure how good or how bad that guess is. That’s the job of the
loss function.

when given the values, the
previous guess, and the results of calculating the errors (or loss) on that guess, can
then generate another one. Over time, its job is to minimize the loss, and by so doing
bring the guessed formula closer and closer to the correct answer.

Source: https://www.youtube.com/watch?v=K20lVDVjPn4
Batch(Hyperparameter): Since one epoch is too big to feed to the computer at once we
divide it in several batches.

Batch_Size: Total number of training examples present in a single batch.

Iterations: are the number of batches needed to complete one epoch

Source: https://telnyx.com/learn-ai/logits-confidence#:~:text=Logits%20are%20a%20neural%20network's%20raw,%20unnormalized%20output,are%20crucial%20for%20transforming%20them%20into%20interpretable%20probabilities.
Logits: are a neural network's raw, unnormalized output values,
typically obtained from the last layer before applying an activation
function such as sigmoid or softmax.

Source: https://www.geeksforgeeks.org/regularization-in-machine-learning/
Regularization is a technique used in machine learning to prevent
overfitting. Overfitting happens when a model learns the training
data too well, including the noise and outliers, which causes it to
perform poorly on new data. In simple terms, regularization adds a
penalty to the model for being too complex, encouraging it to stay
simpler and more general. This way, it’s less likely to make extreme
predictions based on the noise in the data.
 */


public class LogisticRegression {

    public double[] softmax(double[] z) {
        double[] exp = new double[z.length];
        double S = 0;
        // Compute exponentials
        for(int k=0; k<z.length; k++) {
            exp[k] = Math.exp(z[k]);
            // Summation of exponentials
            S += exp[k];
        }
        // Calculate Probabilities
        double[] P = new double[z.length];
        for(int Zk = 0; Zk < z.length; Zk++) {
            P[Zk] = exp[Zk] / S;
        }

        // Print the computed probabilities
//        for (double p:
//             P) {
//            System.out.print(p + " ");
//        }
        return P;
    }

    /*  Source of the equation:
        https://www.geeksforgeeks.org/how-to-implement-softmax-and-cross-entropy-in-python-and-pytorch/

    Calculates the loss or errors the lower the loss score the better */
    //Y; True labels
    public double crossEntropyLoss(double[] y_pred, double[] y_true) {
        double calculatedLoss = 0.0;
        double[] predicted_y = softmax(y_pred);

        for(int y = 0; y < y_true.length; y++) {
            // multiply by -1 to make sure the loss is positive
            calculatedLoss = calculatedLoss + (-1 * y_true[y] * Math.log(predicted_y[y]));
        }

        System.out.printf("Cross entropy loss: %f", calculatedLoss);
        return calculatedLoss;
    }

    //TODO: Implement the L2(Ridge Regression) Regulizer for the loss function

}
