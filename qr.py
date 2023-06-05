import numpy as np

def householder(A):
    (r, c) = np.shape(A)
    Q = np.eye(r, dtype=np.complex128)
    R = np.copy(A)
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
    cnt = 0
    for a in A.T:
        u = np.copy(a)
        for i in range(0, cnt):
            proj = np.dot(np.dot(Q[:, i].T, a), Q[:, i])
            u -= proj
        e = u / np.linalg.norm(u)
        Q[:, cnt] = e
        cnt += 1
    R = Q.T @ A
    return Q, R

def qr_householder(A, iter=500000):
    Ak = np.copy(A)
    n, m = np.shape(Ak)
    QQ = np.eye(n, dtype=np.complex128)
    for _ in range(iter):
        s = Ak.item(n - 1, n - 1)
        smult = s * np.eye(n, dtype=np.complex128)
        Q, R = householder(Ak - smult)
        Ak = R @ Q + smult
        QQ = QQ @ Q
    eig = np.diagonal(Ak)
    print(eig)
    return Ak, QQ

def qr_gram_schmidt(A, iter=500000):
    Ak = np.copy(A)
    n, m = np.shape(Ak)
    QQ = np.eye(n, dtype=np.complex128)
    for _ in range(iter):
        s = Ak[n - 1, n - 1]
        smult = s * np.eye(n, dtype=np.complex128)
        Q, R = gram_schmidt(Ak - smult)
        Ak = R @ Q + smult
        QQ = QQ @ Q
    diagonal = np.diagonal(Ak)
    print(diagonal)
    return Ak, QQ

matrix = np.array([[5, 2, 2], [-8, -3, -4], [4, 2, 3]], dtype=np.complex128)
print("householder")
qr_householder(matrix)
print("gram-schmidt")
qr_gram_schmidt(matrix)
print("build-in")
print(np.linalg.eigvals(matrix))
