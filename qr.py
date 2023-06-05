import numpy as np
def householder_reflection(A):
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
        R = np.dot(Q_cnt, R)
        Q = np.dot(Q, Q_cnt)
    return Q, R

def qr_algorithm(A, iterations=500000):
    Ak = np.copy(A)
    n = np.shape(Ak)[0]
    QQ = np.eye(n, dtype=np.complex128)
    for k in range(iterations):
        s = Ak.item(n - 1, n - 1)
        smult = s * np.eye(n, dtype=np.complex128)
        Q, R = householder_reflection(np.subtract(Ak, smult, dtype=np.complex128))
        Ak = np.add(np.matmul(R, Q), smult)
        QQ = np.matmul(QQ, Q)
    diagonal = np.diagonal(Ak)
    print(diagonal)
    return Ak, QQ


matrix = np.array([[10j, 2, 9], [0.004+2j, 5, 6], [7, 8j, 3]], dtype=np.complex128)
eigenvalues = qr_algorithm(matrix)

# check using build-in function
print(np.linalg.eigvals(matrix))
