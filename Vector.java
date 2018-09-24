class Vector {
  
  public float x, y, z, w;
  private float[] values;
  private int len;
  
  public Vector(int dim) {
    if (dim < 2) {
      return;
    }
    values = new float[dim];
    for (int i = 0; i < dim; i++) {
      values[i] = 0;
    }
    len = dim;
  }
  public Vector(float[] values) {
    if (values.length < 2) {
      return;
    }
    this.values = values;
    len = values.length;
    if (len > 1) {
      x = values[0];
      y = values[1];
    }
    if (len > 2) {
      z = values[2];
    }
    if (len > 3) {
      w = values[3];
    }
  }
  public Vector(float x, float y) {
    this.x = x;
    this.y = y;
    len = 2;
    values = new float[2];
    values[0] = x;
    values[1] = y;
  }
  public Vector(float x, float y, float z) {
    this.x = x;
    this.y = y;
    this.z = z;
    len = 3;
    values = new float[3];
    values[0] = x;
    values[1] = y;
    values[2] = z;
  }
  public Vector(float x, float y, float z, float w) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
    len = 4;
    values = new float[4];
    values[0] = x;
    values[1] = y;
    values[2] = z;
    values[3] = w;
  }
  
  public int length() {
    return len;
  }
  
  public float get(int index) {
    if (index >= len || index < 0) {
      return -9999999;
    } else {
      return values[index];
    }
  }
  
  public void set(float value, int index) {
    if (index >= len || index < 0) {
      return;
    } else {
      values[index] = value;
    }
  }
  
  public float[] toArray() {
    return values;
  }
  
  public int maximum() {
    int index = -1;
    float max_value = -999999999;
    for (int i = 0; i < len; i++) {
      if (values[i] > max_value) {
        max_value = values[i];
        index = i;
      }
    }
    return index;
  }
}
