import numpy as np
import concurrent.futures




def householder(A):
    (r, c) = np.shape(A)
    Q = np.eye(r)
    R = A.copy()
    for cnt in range(r - 1):
        x = R[cnt:, cnt]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)
        if np.isnan(v[0]):
            v = np.zeros_like(v)
        Q_cnt = np.eye(r)
        Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(v, np.conj(v))
        R = Q_cnt @ R
        Q = Q @ Q_cnt
    return Q, R


def qr_householder(A):
    Ak = A.copy()
    n, m = Ak.shape
    QQ = np.eye(n)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for _ in range(50000):
            s = Ak.item(n - 1, n - 1)
            smult = s * np.eye(n)
            futures.append(executor.submit(householder, Ak - smult))
            Ak = futures[-1].result()[1] @ futures[-1].result()[0] + smult
            QQ = QQ @ futures[-1].result()[0]

    eig = np.diagonal(Ak)
    return eig