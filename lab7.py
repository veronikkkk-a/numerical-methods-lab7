import numpy as np


def generate_and_save_data(n=100, filename_a="matrix_a_lab7.txt", filename_b="vector_b_lab7.txt"):
  """Генерує випадкову матрицю А з сильним діагональним переважанням та вектор b."""
  np.random.seed(42)
  A = np.random.rand(n, n) * 10
  # Забезпечуємо суворе діагональне переважання для гарантії збіжності ітераційних методів
  d_sum = np.sum(np.abs(A), axis=1)
  np.fill_diagonal(A, d_sum + 20.0)

  # Задаємо точний розв'язок (усі xi = 2.5)
  x_exact = np.full(n, 2.5)

  # Обчислюємо вектор вільних членів b = A * x_exact
  b = np.dot(A, x_exact)

  # Зберігаємо у файли
  np.savetxt(filename_a, A, fmt="%.6f")
  np.savetxt(filename_b, b, fmt="%.6f")

  print("Дані успішно згенеровано та збережено у файли.")
  return A, b


def read_matrix(filename):
  return np.loadtxt(filename)


def read_vector(filename):
  return np.loadtxt(filename)


def vector_norm(v):
  return np.max(np.abs(v))


def simple_iteration_method(A, b, x0, eps=1e-14, max_iter=1000):
  """Метод простої ітерації."""
  n = A.shape[0]
  norm_a = np.max(np.sum(np.abs(A), axis=1))
  tau = 1.0 / norm_a  # Вибір параметра тау з умови збіжності

  x = x0.copy()
  for k in range(1, max_iter + 1):
    x_new = x - tau * (np.dot(A, x) - b)
    diff = vector_norm(x_new - x)

    if diff < eps:
      print(f"Проста ітерація зійшлася за {k} ітерацій.")
      return x_new, k
    x = x_new
  print("Проста ітерація: досягнуто максимальної кількості ітерацій.")
  return x, max_iter


def jacobi_method(A, b, x0, eps=1e-14, max_iter=1000):
  """Метод Якобі."""
  n = A.shape[0]
  D = np.diag(A)
  R = A - np.diag(D)

  x = x0.copy()
  for k in range(1, max_iter + 1):
    x_new = (b - np.dot(R, x)) / D
    diff = vector_norm(x_new - x)

    if diff < eps:
      print(f"Метод Якобі зійшовся за {k} ітерацій.")
      return x_new, k
    x = x_new
  print("Метод Якобі: досягнуто максимальної кількості ітерацій.")
  return x, max_iter


def seidel_method(A, b, x0, eps=1e-14, max_iter=1000):
  """Метод Зейделя (Гауса-Зейделя)."""
  n = A.shape[0]
  x = x0.copy()

  for k in range(1, max_iter + 1):
    x_new = x.copy()
    for i in range(n):
      s1 = sum(A[i][j] * x_new[j] for j in range(i))
      s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
      x_new[i] = (b[i] - s1 - s2) / A[i][i]

    diff = vector_norm(x_new - x)
    if diff < eps:
      print(f"Метод Зейделя зійшовся за {k} ітерацій.")
      return x_new, k
    x = x_new
  print("Метод Зейделя: досягнуто максимальної кількості ітерацій.")
  return x, max_iter


if __name__ == "__main__":
  # 1. Генерація та збереження
  filename_a = "matrix_a_lab7.txt"
  filename_b = "vector_b_lab7.txt"
  generate_and_save_data(n=100)

  # 2. Зчитування з файлів
  A = read_matrix(filename_a)
  b = read_vector(filename_b)

  # 3. Початкове наближення xi = 1.0
  x0 = np.full(A.shape[0], 1.0)
  eps_target = 1e-14

  print("\n--- Запуск методів ---")
  x_si, iter_si = simple_iteration_method(A, b, x0, eps=eps_target)
  x_jac, iter_jac = jacobi_method(A, b, x0, eps=eps_target)
  x_sei, iter_sei = seidel_method(A, b, x0, eps=eps_target)

  print(f"\nРезультати:")
  print(f"Проста ітерація: ітерацій = {iter_si}")
  print(f"Метод Якобі: ітерацій = {iter_jac}")
  print(f"Метод Зейделя: ітерацій = {iter_sei}")