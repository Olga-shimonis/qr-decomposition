import numpy as np
import concurrent.futures
import time

start = time.time()

def householder(A):
    (r, c) = np.shape(A)
    Q = np.eye(r, dtype=np.complex128)
    R = A.copy()
    for cnt in range(r - 1):
        x = R[cnt:, cnt]
        e = np.zeros_like(x, dtype=np.complex128)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)
        if np.isnan(v[0]):
            v = np.zeros_like(v, dtype=np.complex128)
        Q_cnt = np.eye(r, dtype=np.complex128)
        Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(v, np.conj(v))
        R = Q_cnt @ R
        Q = Q @ Q_cnt
    return Q, R

def gram_schmidt(A):
    Q = np.empty_like(A, dtype=np.complex128)
    column_index = 0
    for a in A.T:
        u = np.copy(a)
        for i in range(0, column_index):
            u -= np.dot(np.dot(Q[:, i].T, a), Q[:, i])
        e = u / np.linalg.norm(u)
        Q[:, column_index] = e
        column_index += 1
    R = Q.T @ A
    return Q, R

def qr_householder(A):
    Ak = A.copy()
    n, m = Ak.shape
    QQ = np.eye(n, dtype=np.complex128)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for _ in range(n - 1):
            s = Ak.item(n - 1, n - 1)
            smult = s * np.eye(n, dtype=np.complex128)
            futures.append(executor.submit(householder, Ak - smult))
            Ak = futures[-1].result()[1] @ futures[-1].result()[0] + smult
            QQ = QQ @ futures[-1].result()[0]

    eig = np.diagonal(Ak)
    return Ak, QQ, eig

def qr_gram_schmidt(A):
    Ak = A.copy()
    n, m = Ak.shape
    QQ = np.eye(n, dtype=np.complex128)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for _ in range(n - 1):
            s = Ak[n - 1, n - 1]
            smult = s * np.eye(n, dtype=np.complex128)
            futures.append(executor.submit(gram_schmidt, Ak - smult))
            Ak = futures[-1].result()[1] @ futures[-1].result()[0] + smult
            QQ = QQ @ futures[-1].result()[0]

    diagonal = np.diagonal(Ak)
    return Ak, QQ, diagonal

matrix = np.array([[5, 2, 2], [-8, -3, -4], [4, 2, 3]], dtype=np.complex128)

print("householder")
Ak, QQ, eig = qr_householder(matrix)
print(eig)

print("gram-schmidt")
Ak, QQ, diagonal = qr_gram_schmidt(matrix)
print(diagonal)

print("built-in")
print(np.linalg.eigvals(matrix)[::-1])

end = time.time() - start
print(end)
