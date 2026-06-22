"""
Matrix Factorization cho gợi ý tin tức (MIND)
  - mỗi user / mỗi item = 1 vector ẩn k chiều  (P[u], Q[i])
  - dự đoán = tích vô hướng:        score(u, i) = p_u · q_i
  - học bằng BÌNH PHƯƠNG SAI SỐ + negative sampling, gradient TỰ TÍNH TAY
    (không autograd, không GPU) — để nhìn rõ "phép màu":
        p_u <- p_u + lr (e q_i - λ p_u)
        q_i <- q_i + lr (e p_u - λ q_i)
"""

import pickle
import numpy as np
from scipy.sparse import load_npz
import config

def train_factors(k, lr, reg, epochs, batch_size, seed=0):
    """Học P, Q từ ma trận thưa user × news. Trả về (P, Q).

    Giống mf_simple_real.py: duyệt các cặp dương (u, i+) theo batch, mỗi cặp
    bốc 1 tin âm i- ngẫu nhiên (tin u chưa đọc), dạy mô hình:
        score(u, i+) -> 1 ,  score(u, i-) -> 0.
    """
    rng = np.random.default_rng(seed)

    matrix = load_npz(str(config.USER_ITEM_MATRIX)).tocoo()
    pos_users = matrix.row.astype(np.int64)   # user của các ô = 1
    pos_items = matrix.col.astype(np.int64)   # tin tương ứng đã đọc
    n_users, n_items = matrix.shape
    n_pairs = len(pos_users)
    print(f"MF train: {n_users} user, {n_items} tin, {n_pairs} tương tác dương")

    # Vector ẩn, khởi tạo ngẫu nhiên nhỏ; gradient descent sẽ nắn dần.
    P = rng.normal(0, 0.1, size=(n_users, k))
    Q = rng.normal(0, 0.1, size=(n_items, k))

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n_pairs)        # xáo trộn thứ tự các cặp dương
        total_sq_err = 0.0

        for start in range(0, n_pairs, batch_size):
            batch = perm[start:start + batch_size]
            u = pos_users[batch]
            i_pos = pos_items[batch]
            i_neg = rng.integers(0, n_items, size=len(batch))  # tin âm ngẫu nhiên

            # Dự đoán (tích vô hướng từng dòng).
            pu = P[u]            # (B, k)
            q_pos = Q[i_pos]     # (B, k)
            q_neg = Q[i_neg]     # (B, k)
            e_pos = 1.0 - (pu * q_pos).sum(axis=1)   # sai số với target 1
            e_neg = 0.0 - (pu * q_neg).sum(axis=1)   # sai số với target 0

            # Gradient tự tính tay (đúng công thức trong mf_simple.py).
            grad_P    = e_pos[:, None] * q_pos + e_neg[:, None] * q_neg - reg * pu
            grad_Qpos = e_pos[:, None] * pu  - reg * q_pos
            grad_Qneg = e_neg[:, None] * pu  - reg * q_neg

            # np.add.at: cộng dồn đúng khi 1 user/tin xuất hiện nhiều lần trong batch.
            np.add.at(P, u,     lr * grad_P)
            np.add.at(Q, i_pos, lr * grad_Qpos)
            np.add.at(Q, i_neg, lr * grad_Qneg)

            total_sq_err += float((e_pos ** 2).sum() + (e_neg ** 2).sum())

        rmse = np.sqrt(total_sq_err / (2 * n_pairs))
        print(f"epoch {epoch:>3}/{epochs}  RMSE = {rmse:.4f}")

    return P, Q

class MatrixFactorizationCF:
    def __init__(self):
        self.Q = None            # (n_items, k) — vector ẩn của từng tin
        self.news2idx = None
        self.idx2news = None

    def fit(self, force_retrain=False):
        """Nạp factors đã cache, hoặc train mới rồi lưu lại."""
        with open(config.NEWS2IDX_PATH, "rb") as f:
            self.news2idx = pickle.load(f)
        self.idx2news = {idx: news_id for news_id, idx in self.news2idx.items()}

        if config.MF_FACTORS_PATH.exists() and not force_retrain:
            data = np.load(config.MF_FACTORS_PATH)
            self.Q = data["Q"]
            print(f"MF: nạp factors từ {config.MF_FACTORS_PATH}  Q={self.Q.shape}")
        else:
            P, Q = train_factors(
                k=config.MF_DIM, lr=config.MF_LR, reg=config.MF_REG,
                epochs=config.MF_EPOCHS, batch_size=config.MF_BATCH_SIZE,
            )
            self.Q = Q
            config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(str(config.MF_FACTORS_PATH), P=P, Q=Q)
            print(f"MF: đã lưu factors vào {config.MF_FACTORS_PATH}")

        return self

    def _profile(self, history):
        """Profile user = trung bình vector ẩn Q của các tin trong history."""
        idx = [self.news2idx[n] for n in history if n in self.news2idx]
        if not idx:
            return None
        return self.Q[idx].mean(axis=0)

    def score_one(self, history, candidates):
        """Chấm điểm candidate cho 1 user. Trả về dict {news_id: score}."""
        return self.score([history], [candidates])[0]

    def score(self, histories, candidate_lists):
        if self.Q is None:
            raise RuntimeError("Phải gọi fit() trước.")

        results = []
        for history, candidates in zip(histories, candidate_lists):
            profile = self._profile(history)
            row = {news_id: 0.0 for news_id in candidates}  # tin lạ / history rỗng -> 0

            if profile is not None:
                for news_id in candidates:
                    j = self.news2idx.get(news_id)
                    if j is not None:
                        row[news_id] = float(profile @ self.Q[j])

            results.append(row)
        return results

def _sanity_check(model, seed=0):
    """Điểm tin ĐÃ ĐỌC nên cao hơn tin NGẪU NHIÊN nếu mô hình học được."""
    rng = np.random.default_rng(seed)
    matrix = load_npz(str(config.USER_ITEM_MATRIX)).tocoo()
    n_items = matrix.shape[1]

    print("\nKiểm tra: điểm TB tin đã đọc vs tin ngẫu nhiên")
    by_user = {}
    for u, i in zip(matrix.row, matrix.col):
        by_user.setdefault(int(u), []).append(int(i))

    for u in list(by_user)[:5]:
        read = np.array(by_user[u])
        rand = rng.integers(0, n_items, size=len(read))
        s_read = float((model.Q[read] @ model.Q[read].mean(axis=0)).mean())
        s_rand = float((model.Q[rand] @ model.Q[read].mean(axis=0)).mean())
        print(f"  user {u:<6}  đã đọc: {s_read:+.3f}   ngẫu nhiên: {s_rand:+.3f}")

if __name__ == "__main__":
    model = MatrixFactorizationCF().fit(force_retrain=True)
    _sanity_check(model)
