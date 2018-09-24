public float[][] matmul(float[][] m1, float[][] m2) {
  int[] m1dim = {m1.length, m1[0].length};
  int[] m2dim = {m2.length, m2[0].length};
  if (m1dim[1] != m2dim[0]) {
    //If the number of columns in m1 != the number of rows in m2
    return null;
  }
  float[][] result = new float[m1dim[0]][m2dim[1]];
  for (int m1_row_index = 0; m1_row_index < m1dim[0]; m1_row_index++) {
    for (int m2_column_index = 0; m2_column_index < m2dim[1]; m2_column_index++) {
      //we will place the result at result[row][column]
      Vector m1_row = new Vector(m1[m1_row_index]);
      Vector m2_column = columnFromMatrix(m2, m2_column_index);
      float value = dotProduct(m1_row, m2_column);
      result[m1_row_index][m2_column_index] = value;
    }
  }
  return result;
}

public Vector matmul(float[][] m1, Vector vec) {
   float[][] columnVector = matmul(m1, vectorToColumnMatrix(vec));
   Vector ret = columnFromMatrix(columnVector, 0);
   return ret;
}

//This is the same thing as a (Nx1) multiplied by (1xN) vector
public float[][] matmul(Vector vec1, Vector vec2) {
  float[] v1 = vec1.toArray();
  float[] v2 = vec2.toArray();
  int M = v1.length;
  int N = v2.length;
  float[][] result = new float[M][N];
  for (int m = 0; m < M; m++) {
    float constant = v1[m];
    float[] row = multiplyByConstant(v2, constant);
    result[m] = row;
  }
  return result;
}

public String vecToString(Vector vec) {
  float[] v = vec.toArray();
  String ret = "";
  int max_num_spaces = 0;
  for (float value : v) {
    if (Float.toString(value).length() > max_num_spaces) {
      max_num_spaces = Float.toString(value).length();
    }
  }
  for (float value : v) {
    int min_num_spaces = Float.toString(value).length();
    String spacing = "";
    for (int i = 0; i < max_num_spaces - min_num_spaces; i++) {
      spacing += " ";
    }
    ret += "[" + spacing + Float.toString(value) + "]";
  }
  return ret;
}

public String matToString(float[][] matrix) {
  String ret = "";
  int max_num_spaces = 0;
  for (float[] line : matrix) {
    for (float value : line) {
      if (Float.toString(value).length() > max_num_spaces) {
        max_num_spaces = Float.toString(value).length();
      }
    }
  }
  for (float[] line : matrix) {
    String line_text = "";
    for (float value : line) {
      int min_num_spaces = Float.toString(value).length();
      String spacing = "";
      for (int i = 0; i < max_num_spaces - min_num_spaces; i++) {
        spacing += " ";
      }
      line_text += "[" + spacing + Float.toString(value) + "]";
    }
    ret += line_text + "\n";
  }
  return ret;
}

public float[] hadamard(float[] v1, float[] v2) {
  if (v1.length != v2.length) {
    return null;
  }
  float[] ret = new float[v1.length];
  for (int i = 0; i < v1.length; i++) {
    ret[i] = v1[i] * v2[i];
  }
  return ret;
}

public Vector hadamard(Vector v1, Vector v2) {
  Vector ret = new Vector(hadamard(v1.toArray(), v2.toArray()));
  return ret;
}

public float[][] hadamard(float[][] m1, float[][] m2) {
  if (m1.length != m2.length || m1[0].length != m2[0].length) {
    return null;
  }
  float[][] ret = new float[m1.length][m1[0].length];
  for (int m = 0; m < m1.length; m++) {
    for (int n = 0; n < m1[0].length; n++) {
      ret[m][n] = m1[m][n] * m2[m][n];
    }
  }
  return ret;
}

public float[] subtraction(float[] v1, float[] v2) {
  float[] v2_neg = multiplyByConstant(v2, -1);
  float[] result = addition(v1, v2_neg);
  return result;
}

public Vector subtraction(Vector v1, Vector v2) {
  float[] result = subtraction(v1.toArray(), v2.toArray());
  Vector ret = new Vector(result);
  return ret;
}

public float[][] subtraction(float[][] m1, float[][] m2) {
  float[][] m2_neg = multiplyByConstant(m2, -1);
  float[][] result = addition(m1, m2_neg);
  return result;
}

public float[][] addition(float[][] m1, float[][] m2) {
  if (m1.length != m2.length || m1[0].length != m2[0].length) {
    return null;
  }
  float[][] ret = new float[m1.length][m1[0].length];
  for (int m = 0; m < m1.length; m++) {
    for (int n = 0; n < m1[0].length; n++) {
      ret[m][n] = m1[m][n] + m2[m][n];
    }
  }
  return ret;
}

public float[] addition(float[] v1, float[] v2) {
  if (v1.length != v2.length) {
    return null;
  }
  float[] ret = new float[v1.length];
  for (int i = 0; i < v1.length; i++) {
    ret[i] = v1[i] + v2[i];
  }
  return ret;
}

public float magnitude(float[] v) {
  float sum = 0;
  for (float value : v) {
    sum += value * value;
  }
  float magnitude = sqrt(sum);
  return magnitude;
}

public float magnitude(Vector v) {
  return magnitude(v.toArray());
}

public Vector addition(Vector v1, Vector v2) {
  Vector ret = new Vector(addition(v1.toArray(), v2.toArray()));
  return ret;
}

public float[][] vectorToColumnMatrix(Vector vec) {
  float[] values = vec.toArray();
  float[][] colmat = new float[values.length][1];
  for (int i = 0; i < values.length; i++) {
    colmat[i][0] = values[i];
  }
  return colmat;
}

public float[][] multiplyByConstant(float[][] matrix, float constant) {
  float[][] new_matrix = new float[matrix.length][matrix[0].length];
  for (int m = 0; m < matrix.length; m++) {
    for (int n = 0; n < matrix[0].length; n++) {
      new_matrix[m][n] = matrix[m][n] * constant;
    }
  }
  return new_matrix;
}

public Vector multiplyByConstant(Vector vec, float constant) {
  float[] result = multiplyByConstant(vec.toArray(), constant);
  Vector ret = new Vector(result);
  return ret;
}

public float[] multiplyByConstant(float[] v, float constant) {
  float[] result = new float[v.length];
  for (int i = 0; i < v.length; i++) {
    result[i] = v[i] * constant;
  }
  return result;
}

public float[][] transpose(float[][] matrix) {
  int m = matrix.length;
  int n = matrix[0].length;
  float[][] transposed = new float[n][m];
  for (int x = 0; x < n; x++) {
    for (int y = 0; y < m; y++) {
      transposed[x][y] = matrix[y][x];
    }
  }
  return transposed;
}

public float[] toExponent(float[] v, float exponent) {
  float[] ret = new float[v.length];
  for (int i = 0; i < v.length; i++) {
    ret[i] = pow(v[i], exponent);
  }
  return ret;
}

public Vector toExponent(Vector vec, float exponent) {
  float[] result = toExponent(vec.toArray(), exponent);
  Vector ret = new Vector(result);
  return ret;
}

//Takes each value as an exponent to the constant given
//i.e. (constant)^(vector)
public Vector asExponentTo(float constant, Vector vec) {
  float[] result = asExponentTo(constant, vec.toArray());
  Vector ret = new Vector(result);
  return ret;
}

//Takes each value as an exponent to the constant given
//i.e. (constant)^(vector)
public float[] asExponentTo(float constant, float[] v) {
  float[] ret = new float[v.length];
  for (int i = 0; i < v.length; i++) {
    ret[i] = pow(constant, v[i]);
 
  }
  return ret;
}

public Vector columnFromMatrix(float[][] matrix, int column_index) {
  if (column_index >= matrix[0].length || column_index < 0) {
    return null;
  }
  float[] column = new float[matrix.length];
  for (int i = 0; i < matrix.length; i++) {
    column[i] = matrix[i][column_index];
  }
  Vector ret = new Vector(column);
  return ret;
}

public float dotProduct(float[] v1, float[] v2) {
  if (v1.length != v2.length) {
    return -99999999;
  }
  float sum = 0;
  for (int i = 0; i < v1.length; i++) {
    sum += v1[i] * v2[i];
  }
  return sum;
}

public float dotProduct(Vector v1, Vector v2) {
  return dotProduct(v1.toArray(), v2.toArray());
}
