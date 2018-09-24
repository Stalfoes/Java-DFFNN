import java.util.Random;
import java.lang.Math;

class Network {
  
  /*
   * Hello and welcome. It's that time of year I reprogram my neural network into Java.
   * The entire point of this is so I have a way I can use it for more 2D games.
   * Not that Python can't do 2D. It's just this seems easier for now.
   * Maybe I'll come back and do more with the Python version of this.
   *
   * Anyways, this class is the same as in Python hopefully. We're just remaking it in Java.
   * Since the other day I programmed matrix operations and vector operations.
   * That should give us what we need in order to make this happen.
   * Without further ado, let's hop right into it.
   *
   * I guess I should mention it's a deep feed-forward neural network.
   *
   * Important resources:
   *   Michael Nielsen's FANTASTIC book on neural networks and deep learning:
   *     http://neuralnetworksanddeeplearning.com/
   *     I can't stress enough how great this book is. All the math is proved, and lots of example code it given.
   *   3Brown1Blue's YouTube channel:
   *     https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw
   *     This is helpful when trying to visualize gradient descent and some of the more complicated math bits.
   *     He has a series on neural networks and deep learning. It's not done, but the stuff he has there already
   *       is very good.
   */

  private int[] sizes; //The layer's sizes.
  private int L; //The length of the neural network. N layers
  private float[][][] weightsMatrix; //An array with matrices inside.
  private Vector[] biasesMatrix; //An array with vectors inside
  private Vector[] activationsMatrix; //An array with vectors inside.
  private Vector[] Zs; //Contains the used activations
  private Vector[] errorsMatrix; //Contains the errors over one epoch
  private int num_correct_in_epoch = 0; //Keeps track of how many we have correct in each epoch

  private ArrayList<Vector[]> errors_over_epoch; //This stores each 'errorsMatrix' for each epoch
  private ArrayList<Vector[]> activations_over_epoch; //This stores each 'activationsMatrix' for each epoch
  private ArrayList<Vector[]> epoch_summary; //This stores the 'batch_triple'

  private ArrayList<Boolean> epoch_summary_bool; //This does the same as epoch_summary but keeps track of whether we got an example correct or not

  private Random random = new Random(); //Used to compute gaussian-distributed weights and biases.

  /**
   * The constructor for the Network class. This creates random weights and biases
   *   based on the sizes array. These are of gausian distribution.
   * Sizes is an array of integers, with the layer's sizes of neurons.
   */
  public Network(int[] sizes) {

    this.sizes = sizes;
    L = sizes.length;

    weightsMatrix = new float[L - 1][][];

    //The main loop generating EACH matrix
    for (int i = 0; i < L - 1; i++) {
      //Generate the guassian values for each spot in the matrix
      int M = sizes[i+1];
      int N = sizes[i];
      float[][] matrix = new float[M][N];
      for (int m = 0; m < M; i++) {
        for (int n = 0; n < N; i++) {
          float value = (float) random.nextGaussian();
          matrix[m][n] = value;
        }
      }
      weightsMatrix[i] = matrix;
    }

    biasesMatrix = new Vector[L];

    //The main loop generating EACH vector
    for (int i = 0; i < L; i++) {
      int M = sizes[i];
      float[] vector_values = new float[M];
      //Generate the gaussian values for each spot in the vector
      for (int m = 0; m < M; m++) {
        float value = (float) random.nextGaussian();
        vector_values[m] = value;
      }
      Vector bias = new Vector(vector_values);
      biasesMatrix[i] = bias;
    }

    activationsMatrix = new Vector[L];
    errorsMatrix = new Vector[L];
    Zs = new Vector[L - 1];

    errors_over_epoch = new ArrayList<Vector[]>();
    activations_over_epoch = new ArrayList<Vector[]>();
    epoch_summary = new ArrayList<Vector[]>();

    epoch_summary_bool = new ArrayList<Boolean>();
  }

  /**
   * This method allows us to set the weights and biases of the neural network.
   * This method is specifically designed to allow us to use the network in part with
   *   a genetic algorithm (or something like it).
   */
  public void setWeightsAndBiases(float[][][] weights, Vector[] biases) {
    weightsMatrix = weights;
    biasesMatrix = biases;
  }

  //To be honest I'm not sure I need the refresh epoch method. We'll see.
  //I've decided it's a good idea


  /**
   * This method resets all the variables that are no longer needed. This is generally called
   *   per epoch, in order to reset the variables that are needed for the next epoch.
   * It is also called in inputOutput(...) to clear the variables for use.
   */
  private void refreshEpoch() {
    activationsMatrix = new Vector[L];
    errorsMatrix = new Vector[L];
    Zs = new Vector[L - 1];

    errors_over_epoch = new ArrayList<Vector[]>();
    activations_over_epoch = new ArrayList<Vector[]>();
    epoch_summary = new ArrayList<Vector[]>();

    epoch_summary_bool = new ArrayList<Boolean>();

    num_correct_in_epoch = 0;
  }

  /**
   * This method is where we do all the learning with the network. We don't have
   *   stochastic gradient descent sadly. I never programmed that as I deemed it
   *   unecessary at the time of programming this in Python.
   * Parameters:
   *   int num_epochs
   *     This is the number of times to run through and train off of the training_data.
   *     The higher the number, the longer the training takes.
   *   Vector[][] training_data
   *     This is a special parameter. The formatting is very important.
   *     The format is:
   *       training_data[n] = [Vector input, Vector expected_output]
   *     The input Vector is the activations of the first layer, and we train off of
   *       the expected output vector.
   *     The length of the Vectors should be equal to the sizes of the first and last layers.
   *   float eta
   *     Eta is the greek letter that looks like an "n".
   *     This is the learning rate of the network. It tells the network how fast to learn.
   *     Finding the optimal eta is something that has to be done manually or through the
   *       use of a genetic algorithm.
   *   String output_type
   *     This controls if we want the output of the network to happen or not.
   *     Meaning output into the console, to log the progression.
   *     Acceptable values:
   *       'all' = All progression is logged.
   *       'last' = Only the final epoch is logged.
   *       'none' = None of the progression is logged.
   */
  public void learn(int num_epochs, Vector[][] training_data, float eta, String output_type) {

    for (int epoch = 0; epoch < num_epochs; epoch++) {

      refreshEpoch();

      training_data = shuffleData(training_data); //Shuffle the training data

      for (int training_data_index = 0; training_data_index < training_data.length; training_data_index++) {

        Vector[] batch_pair = training_data[training_data_index];

        activationsMatrix[0] = batch_pair[0];
        for (int s = 1; s < sizes.length; s++) {
          float[] zeros = new float[sizes[s]];
          Zs[s - 1] = new Vector(zeros);
        }

        feedforward();

        for (int s = 1; s < sizes.length; s++) {
          float[] zeros = new float[sizes[s]];
          errorsMatrix[s - 1] = new Vector(zeros);
        }

        Vector Y = batch_pair[1];

        Vector lastActivation = activationsMatrix[activationsMatrix.length - 1];
        Vector costDer = costDerivative(lastActivation, Y);

        Vector left = sigmoid_prime(Zs[Zs.length - 1]);
        errorsMatrix[errorsMatrix.length - 1] = hadamard(costDer, left);

        backpropogate();

        errors_over_epoch.add(errorsMatrix);
        activations_over_epoch.add(activationsMatrix);

        evaluateSingle(batch_pair);
      }

      gradientDescent(eta, training_data, epoch);

      evaluateEpoch(epoch, training_data.length, output_type, num_epochs);
    }
  }

  /**
   * This method is meant to output the network's progress while training is occuring.
   * It's a bit inefficient but I made this class EXACTLY the same as I did in Python.
   * I can make it better later if speed seems to be an issue (or if I feel like it :P).
   */
  private void evaluateEpoch(int epoch, int num_trained, String output_type, int n_epochs) {

    float performance = (float) num_correct_in_epoch / (float) num_trained;

    if (output_type.equals("all") || (output_type.equals("last") && epoch == n_epochs - 1)) {
      String output = String.format("Epoch %d: %.5f => %d / %d", epoch + 1, performance, num_correct_in_epoch, num_trained);
      print(output);
    }
  }

  /**
   * This performs the technique called "gradient descent".
   * It is the technique used to change the network's weights and biases based
   *   on the performance of the network as it trains.
   * The equations can generally be found online, and they require a good knowledge of
   *   linear algebra and multi-variable calculus to understand.
   * I can try to explain it though:
   *  - We get a constant given by eta and the length of the training data. The length of the training data
   *     set allows the constant to be less if we have more training examples. This is to lower the "impact" of
   *     each example on the outcome of the training. So that no example changes the network too much.
   *     The constant is negative as we want to descend the gradient to lower the cost / improve accuracy.
   * - Gradient Descent is the process of going down the cost function to find minimums. This is like using the slope
   *     of a function to find a minimum. This minimum on the cost function is the place with the least "cost" or the most
   *     accuracy, which is where we want to end up. With multivariable calculus or rather functions in higher dimensions
   *     this slope is kind of like the "gradient". The gradient is defined as the direction as to increase the fastest, or
   *     rather the easiest way to a maximum. So to find the fastest way to a minimum, we want the negative gradient.
   * - We go through each set of weights and biases for each layer, and change them based on some equations.
   *     These equations are difficult to define in just plain text because they're kind of complicated.
   * - For the weights:
   *    - We create a change in weights by doing:
   *        (change in weights) = (constant) * [ (errors)^T * (activations) ]
   *    - Keep in mind these errors and activations are over the entire epoch. So we train the network only after
   *        each epoch.
   * - The biases are about the same, but the change in biases are just equal to the errors.
   * This will modify all our weights and biases according to the training that has been done, in order to descend
   *   the gradient, and therefore improve the network.
   */

  private void gradientDescent(float eta, Vector[][] training_data, int epoch) {

    float multiplier = -eta / training_data.length;

    for (int i = 0; i < L - 1; i++) {

      float[][] beforeWeights = weightsMatrix[i];

      float[][] weightsSummation = new float[beforeWeights.length][beforeWeights[i].length];

      for (int x = 0; x < training_data.length; x++) {

        Vector errors = errors_over_epoch.get(x)[i];
        Vector activations = activations_over_epoch.get(x)[i];

        float[][] ds = matmul(errors, activations);
        weightsSummation = addition(weightsSummation, ds);
      }

      float[][] changeInWeights = multiplyByConstant(weightsSummation, multiplier);
      float[][] afterWeights = addition(beforeWeights, changeInWeights);
      weightsMatrix[i] = afterWeights;

      Vector beforeBiases = biasesMatrix[i];

      float[] zeroes = new float[beforeBiases.length()];

      Vector biasesSummation = new Vector(zeroes);

      for (int x = 0; i < training_data.length; i++) {

        Vector ds = errors_over_epoch.get(x)[i];
        biasesSummation = addition(biasesSummation, ds);
      }

      Vector changeInBiases = multiplyByConstant(biasesSummation, multiplier);
      Vector afterBiases = addition(beforeBiases, changeInBiases);
      biasesMatrix[i] = afterBiases;
    }
  }

  /**
   * This method takes the batch_pair of inputs and expected outputs and compares
   *   what the network has outputted vs. what the network is expected to output.
   * We add to the count of num_correct_in_epoch if we get the example correct, and
   *   store the necessary information about the comparison in the necessary variables.
   */
  private void evaluateSingle(Vector[] batch_pair) {

    Vector last_activations = activationsMatrix[activationsMatrix.length - 1];

    Vector theoretical_activations = batch_pair[1];

    int max_index_theoretical = theoretical_activations.maximum();
    int max_index_experimental = last_activations.maximum();

    boolean maxes_match = max_index_theoretical == max_index_experimental;

    if (maxes_match) {
      num_correct_in_epoch++;
    }

    Vector[] batch_triple = new Vector[3];
    batch_triple[0] = batch_pair[0];
    batch_triple[1] = batch_pair[1];
    batch_triple[2] = last_activations;

    epoch_summary.add(batch_triple);
    epoch_summary_bool.add(maxes_match);
  }

  /**
   * This method backpropogates through the network in order to compute the errors of the neural network.
   * These errors help us perform the gradient descent.
   */
  private void backpropogate() {
    for (int i = L - 2; i > 2; i--) {
      float[][] weights_T = transpose(weightsMatrix[i + 1]);
      Vector errors = errorsMatrix[i + 1];
      Vector left_side = matmul(weights_T, errors);
      Vector right_side = sigmoid_prime(Zs[i]);
      Vector result = hadamard(left_side, right_side);
      errorsMatrix[i] = result;
    }
  }

  /**
   * This is the derivative of the cost function.
   * I am using the "Mean Squared" cost function for this neural network, but there are
   *   other cost functions that I've been told work better for training a network.
   * -> dC(mean-squared) = A - Y
   */
  private Vector costDerivative(Vector A, Vector Y) {
    Vector ret = subtraction(A, Y);
    return ret;
  }

  /**
    * This is the feedforward method. We simply go through the neural network and 
    *   compute the Z's for each layer. This is the process of just going from
    *   the initial activations to the final activations.
    * Here we also add to the Zs vector so we can store that information.
    * To compute the z vector for a given activation vector, a:
    *
    *   z = W * a + b
   */
  private void feedforward() {

    for (int i = 0; i < L - 1; i++) {

      Vector w_a = matmul(weightsMatrix[i], activationsMatrix[i]);
      Zs[i] = addition(w_a, biasesMatrix[i]);

      Vector nextActivation = sigmoid(Zs[i]);

      activationsMatrix[i + 1] = nextActivation;
    }
  }

  /**
    * This is one of the only public methods for the network.
    * This method takes in some input activations and returns the fed-forward
    *   output activations. This is the equivalent of just doing a feed-forward
    *   from given input activations.
    * The reason we need to separate these two is due to how I've set up my variables
    *   and methods.
    * This method returns a vector of output_activations.
   */

  public Vector inputOutput(Vector input_activations) {
    refreshEpoch();
    activationsMatrix[0] = input_activations;
    for (int s = 1; s < sizes.length - 1; s++) {
      float[] zeros = new float[s];
      Zs[s - 1] = new Vector(zeros);
    }
    feedforward();
    return activationsMatrix[activationsMatrix.length - 1];
  }

  //public Vector, boolean inputOutput(Vector input_activations, Vector expected_output_activations)

  /**
    * Another one of the only public methods.
    * This is nearly the same as the other method of the same name.
    * (To be honest, I don't know of a better name for these methods, but I'm sure there are)
    * This method takes in input_activaions and output_activations and returns a boolean as to whether
    *   the network is able to get the proper output activations; or rather, we test to see if the
    *   network got the example correct, based on its values.
    * The Python equivalent also returns the Vector of computed output activations, but on here we can
    *   just use the other inputOutput method to get those if need be.
   */
  public boolean inputOutput(Vector input_activations, Vector expected_output_activations) {
    refreshEpoch();
    activationsMatrix[0] = input_activations;
    for (int s = 1; s < sizes.length - 1; s++) {
      float[] zeros = new float[s];
      Zs[s - 1] = new Vector(zeros);
    }
    feedforward();
    Vector[] evalSingArr = {input_activations, expected_output_activations};
    evaluateSingle(evalSingArr);
    if (num_correct_in_epoch == 1) {
      return true;
    } else {
      return false;
    }
  }
}


/**
  * This method takes in a vector 'z' and computes the sigmoid function applied to 'z'.
  * We return the computer vector.
 */
Vector sigmoid(Vector z) {
  float[] ones_f = new float[z.length()];
  for (int i = 0; i < z.length(); i++) {
    ones_f[i] = 1;
  }
  Vector ones = new Vector(ones_f);

  Vector negative_z = multiplyByConstant(z, -1);
  Vector e_to_neg_z = asExponentTo((float) Math.E, negative_z);
  Vector denominator = addition(ones, e_to_neg_z);
  Vector ret = toExponent(denominator, -1);

  return ret;
}

/**
  * This is the derivative of the sigmoid function.
 */
Vector sigmoid_prime(Vector z) {
  float[] ones_f = new float[z.length()];
  for (int i = 0; i < z.length(); i++) {
    ones_f[i] = 1;
  }
  Vector ones = new Vector(ones_f);

  Vector left = sigmoid(z);
  Vector right = subtraction(ones, left);
  Vector ret = hadamard(left, right);

  return ret;
}

/**
  * This method shuffles a matrix of Vectors called 'data'.
  * This method was purely designed to shuffle the training data.
  * It shuffles the data N times where N is the length of the training data.
  * We can adjust it to shuffle more if need be.
 */
Vector[][] shuffleData(Vector[][] data) {
  Random random = new Random();
  int shuffle_amount = data.length; //amount of times we shuffle
  for (int i = 0; i < shuffle_amount; i++) {
    //We actually just swap around elements randomly
    int randPos = random.nextInt(data.length);
    Vector[] pair = data[i];
    data[i] = data[randPos];
    data[randPos] = pair;
  }

  return data;
}
