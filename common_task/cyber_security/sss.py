import random
from typing import List
import numpy as np
from dataclasses import dataclass

@dataclass
class Share:
    x: int
    y: int

class ShamirSecretSharing:
    def __init__(self, prime: int = 2**31 - 1):
        self.prime = prime

    def _generate_polynomial(self, secret: int, threshold: int) -> List[int]:
        coefficients = [secret]
        for _ in range(threshold - 1):
            coefficients.append(random.randint(0, self.prime - 1))
        return coefficients

    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        result = 0
        for coef in reversed(coefficients):
            result = (result * x + coef) % self.prime
        return result

    def split_secret(self, secret: int, n_shares: int, threshold: int) -> List[Share]:
        if threshold > n_shares:
            raise ValueError("Threshold cannot be greater than number of shares")
        coefficients = self._generate_polynomial(secret, threshold)
        shares = []
        for i in range(1, n_shares + 1):
            y = self._evaluate_polynomial(coefficients, i)
            shares.append(Share(x=i, y=y))
        return shares

    def reconstruct_secret(self, shares: List[Share]) -> int:
        if not shares:
            raise ValueError("No shares provided")
        def lagrange_basis(j: int, x: int) -> int:
            numerator = denominator = 1
            for m in range(len(shares)):
                if m != j:
                    numerator = (numerator * (x - shares[m].x)) % self.prime
                    denominator = (denominator * (shares[j].x - shares[m].x)) % self.prime
            denominator_inv = pow(denominator, -1, self.prime)
            return (numerator * denominator_inv) % self.prime
        secret = 0
        for j in range(len(shares)):
            basis = lagrange_basis(j, 0)
            secret = (secret + (shares[j].y * basis) % self.prime) % self.prime
        return secret

def main():
    sss = ShamirSecretSharing()
    secret = 123456789
    shares = sss.split_secret(secret, n_shares=5, threshold=3)
    print(f"Original secret: {secret}")
    print("\nGenerated shares:")
    for i, share in enumerate(shares, 1):
        print(f"Share {i}: ({share.x}, {share.y})")
    selected_shares = shares[:3]
    reconstructed_secret = sss.reconstruct_secret(selected_shares)
    print(f"\nReconstructed secret using 3 shares: {reconstructed_secret}")
    print(f"Reconstruction successful: {secret == reconstructed_secret}")

if __name__ == "__main__":
    main()
