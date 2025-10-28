"""
Tests for rebalancing strategies.

Tests cover:
1. Asymmetric rebalancing with 100% relative bands
2. Rebalance counting only on sell transactions
3. Contribution allocation strategies
"""
import pytest
import numpy as np
from simulation_engine import _simulate_portfolio_numba, _allocate_contribution_asymmetric, _check_asymmetric_rebalance_needed


class TestAsymmetricContributionAllocation:
    """Test the asymmetric contribution allocation strategy."""
    
    def test_allocation_to_below_low_band_asset(self):
        """Test that contributions prioritize assets below low band (target/2)."""
        # Portfolio: 60% VOO, 40% GOVT
        target_weights = np.array([0.6, 0.4])
        
        # Current holdings: VOO is at 25% (below low band of 30%), GOVT is at 75%
        # Low band for VOO: 0.6 / 2 = 0.3 (30%)
        # VOO at 25% is BELOW low band - should get priority
        holdings_value = np.array([25000.0, 75000.0])  # Total: $100k
        contrib = 1000.0
        
        allocation = _allocate_contribution_asymmetric(holdings_value, target_weights, contrib)
        
        # VOO is below low band, so should get majority of contribution
        assert allocation[0] > 500.0, "Asset below low band should get majority of contribution"
        assert np.sum(allocation) == pytest.approx(contrib, abs=1e-6)
    
    def test_allocation_when_all_in_band(self):
        """Test allocation when all assets are within bands but below target."""
        # Portfolio: 60% VOO, 40% GOVT
        target_weights = np.array([0.6, 0.4])
        
        # Current holdings: VOO at 50% (in band: 30%-120%, below target 60%)
        # GOVT at 50% (in band: 20%-80%, above target 40%)
        holdings_value = np.array([50000.0, 50000.0])  # Total: $100k
        contrib = 1000.0
        
        allocation = _allocate_contribution_asymmetric(holdings_value, target_weights, contrib)
        
        # VOO is below target, should get more
        assert allocation[0] > allocation[1], "Asset below target should get more contribution"
        assert np.sum(allocation) == pytest.approx(contrib, abs=1e-6)
    
    def test_allocation_when_all_at_target(self):
        """Test allocation when all assets are at target."""
        target_weights = np.array([0.6, 0.4])
        holdings_value = np.array([60000.0, 40000.0])  # Exactly at target
        contrib = 1000.0
        
        allocation = _allocate_contribution_asymmetric(holdings_value, target_weights, contrib)
        
        # Should allocate proportionally to target weights
        assert allocation[0] == pytest.approx(600.0, abs=1.0)
        assert allocation[1] == pytest.approx(400.0, abs=1.0)
        assert np.sum(allocation) == pytest.approx(contrib, abs=1e-6)
    
    def test_multiple_assets_below_low_band(self):
        """Test allocation when multiple assets are below low band."""
        # 3-asset portfolio
        target_weights = np.array([0.5, 0.3, 0.2])
        
        # Both first two assets below low band
        # Low bands: 0.25, 0.15, 0.10
        holdings_value = np.array([20000.0, 10000.0, 70000.0])  # 20%, 10%, 70%
        contrib = 1000.0
        
        allocation = _allocate_contribution_asymmetric(holdings_value, target_weights, contrib)
        
        # Both underweight assets should get contributions
        assert allocation[0] > 0
        assert allocation[1] > 0
        # Overweight asset should get little/no contribution
        assert allocation[2] < allocation[0]
        assert np.sum(allocation) == pytest.approx(contrib, abs=1e-6)


class TestAsymmetricRebalanceCheck:
    """Test the asymmetric rebalance check (high band detection)."""
    
    def test_no_rebalance_when_in_band(self):
        """Test that no rebalance is triggered when all assets are in band."""
        target_weights = np.array([0.6, 0.4])
        
        # VOO at 70% (below high band of 120%), GOVT at 30%
        holdings_value = np.array([70000.0, 30000.0])
        
        needs_rebalance = _check_asymmetric_rebalance_needed(holdings_value, target_weights)
        assert not needs_rebalance, "Should not rebalance when all assets are in band"
    
    def test_rebalance_when_exceeds_high_band(self):
        """Test that rebalance is triggered when asset exceeds high band (target*2)."""
        # Use a smaller target weight to make exceeding feasible
        target_weights = np.array([0.3, 0.7])
        
        # First asset at 65% of portfolio exceeds high band of 60% (30%*2)
        # High band for first asset: 0.3 * 2 = 0.6 (60% of portfolio)
        holdings_value = np.array([65000.0, 35000.0])  # 65% / 35%
        
        needs_rebalance = _check_asymmetric_rebalance_needed(holdings_value, target_weights)
        assert needs_rebalance, "Should rebalance when asset exceeds high band"
    
    def test_boundary_case_exactly_at_high_band(self):
        """Test boundary case when asset is exactly at high band."""
        target_weights = np.array([0.5, 0.5])
        
        # First asset at exactly 100% (high band = 50% * 2 = 100%)
        # This means first asset is 100% of portfolio
        holdings_value = np.array([100000.0, 0.0])
        
        needs_rebalance = _check_asymmetric_rebalance_needed(holdings_value, target_weights)
        # At boundary, should not trigger (we use > not >=)
        # Actually, 100% > 100% is false, so no rebalance
        # But current_weight = 100000/100000 = 1.0, high_band = 0.5*2 = 1.0
        # So current_weight > high_band means 1.0 > 1.0 which is False
        assert not needs_rebalance, "Should not rebalance exactly at high band"


class TestRebalanceCountingOnSells:
    """Test that rebalances are only counted when selling occurs."""
    
    def test_standard_rebalancing_counts_sells_only(self):
        """Test that standard threshold rebalancing only counts when selling."""
        target_weights = np.array([0.5, 0.5])
        
        # Create scenario where one asset grows significantly
        # Need 2 months: first month creates drift, second month triggers rebalance
        monthly_returns = np.array([
            [0.5, 0.0],  # Month 1: First asset +50%, creates drift
            [0.0, 0.0],  # Month 2: No growth, but rebalance happens at start
        ])
        
        initial_investment = 100000
        rebalance_threshold = 0.05
        num_months = 2
        contrib = 0  # No contributions to isolate rebalancing
        
        values, rebalance_count, _ = _simulate_portfolio_numba(
            target_weights, monthly_returns, contrib,
            initial_investment, rebalance_threshold, num_months,
            asymmetric_rebalancing=False
        )
        
        # After month 1: ~$75k in first, ~$50k in second (60%/40%)
        # At start of month 2: rebalance triggers (deviation > 5%)
        # Rebalance count should be 1 (we sell first asset to buy second)
        assert rebalance_count == 1, "Should count rebalance when selling occurs"
    
    def test_no_rebalance_count_when_only_buying(self):
        """Test that adding contributions doesn't count as rebalance."""
        target_weights = np.array([0.5, 0.5])
        
        # No returns, just contributions
        monthly_returns = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        
        initial_investment = 100000
        rebalance_threshold = 0.05
        num_months = 2
        contrib = 5000  # Large contribution
        
        values, rebalance_count, _ = _simulate_portfolio_numba(
            target_weights, monthly_returns, contrib,
            initial_investment, rebalance_threshold, num_months,
            asymmetric_rebalancing=False
        )
        
        # Contributions are allocated to maintain balance, but no selling occurs
        assert rebalance_count == 0, "Should not count rebalances when only buying"
    
    def test_asymmetric_rebalancing_counts_sells_only(self):
        """Test that asymmetric rebalancing only counts when selling."""
        # Use smaller target to make exceeding high band feasible
        target_weights = np.array([0.25, 0.75])
        
        # Create scenario where first asset exceeds high band (50% = 25%*2)
        # Need 2 months: first creates LARGE drift, second triggers rebalance
        monthly_returns = np.array([
            [3.0, 0.0],  # Month 1: First asset quadruples (300% return)
            [0.0, 0.0],  # Month 2: No growth, rebalance happens
        ])
        
        initial_investment = 100000
        rebalance_threshold = 0.05  # Not used in asymmetric mode
        num_months = 2
        contrib = 0
        
        values, rebalance_count, _ = _simulate_portfolio_numba(
            target_weights, monthly_returns, contrib,
            initial_investment, rebalance_threshold, num_months,
            asymmetric_rebalancing=True
        )
        
        # After month 1: First asset at $100k ($25k * 4), second at $75k
        # Total: $175k, weights: 57.1% / 42.9%
        # First asset at 57.1% EXCEEDS high band of 50% (25%*2)
        # At start of month 2: asymmetric rebalance should trigger
        assert rebalance_count == 1, "Should count asymmetric rebalance when selling"


class TestAsymmetricVsStandardRebalancing:
    """Compare asymmetric and standard rebalancing strategies."""
    
    def test_asymmetric_has_fewer_rebalances(self):
        """Test that asymmetric rebalancing triggers less often than standard."""
        target_weights = np.array([0.6, 0.4])
        
        # Create returns that cause moderate drift
        np.random.seed(42)
        monthly_returns = np.random.normal(0.01, 0.05, size=(120, 2))
        
        initial_investment = 100000
        rebalance_threshold = 0.05
        num_months = 120
        contrib = 1000
        
        # Run with standard rebalancing
        _, standard_count, _ = _simulate_portfolio_numba(
            target_weights, monthly_returns, contrib,
            initial_investment, rebalance_threshold, num_months,
            asymmetric_rebalancing=False
        )
        
        # Run with asymmetric rebalancing
        _, asymmetric_count, _ = _simulate_portfolio_numba(
            target_weights, monthly_returns, contrib,
            initial_investment, rebalance_threshold, num_months,
            asymmetric_rebalancing=True
        )
        
        # Asymmetric should have fewer rebalances (wider bands)
        assert asymmetric_count <= standard_count, \
            f"Asymmetric ({asymmetric_count}) should have <= rebalances than standard ({standard_count})"
    
    def test_asymmetric_allows_drift_within_bands(self):
        """Test that asymmetric rebalancing allows drift within wide bands."""
        target_weights = np.array([0.5, 0.5])
        
        # One asset grows significantly (within asymmetric bands but outside standard threshold)
        # Need 2 months: first creates drift, second checks rebalancing
        monthly_returns = np.array([
            [0.30, 0.0],  # Month 1: +30% for first asset
            [0.0, 0.0],   # Month 2: No growth, rebalance check happens
        ])
        
        initial_investment = 100000
        rebalance_threshold = 0.05  # 5% threshold
        num_months = 2
        contrib = 0
        
        # Run with standard rebalancing
        _, standard_count, standard_holdings = _simulate_portfolio_numba(
            target_weights, monthly_returns, contrib,
            initial_investment, rebalance_threshold, num_months,
            asymmetric_rebalancing=False
        )
        
        # Run with asymmetric rebalancing
        _, asymmetric_count, asymmetric_holdings = _simulate_portfolio_numba(
            target_weights, monthly_returns, contrib,
            initial_investment, rebalance_threshold, num_months,
            asymmetric_rebalancing=True
        )
        
        # After month 1: $65k/$50k = 56.5%/43.5% (deviation of 6.5% > 5% threshold)
        # Standard should rebalance in month 2
        # Asymmetric should NOT (56.5% is within 25%-100% band)
        assert standard_count >= 1, "Standard should have rebalanced"
        assert asymmetric_count == 0, "Asymmetric should NOT have rebalanced"
        
        # Check that holdings are different
        assert not np.allclose(standard_holdings, asymmetric_holdings), \
            "Holdings should differ between strategies"
    
    def test_both_strategies_converge_with_no_drift(self):
        """Test that both strategies produce similar results with no drift."""
        target_weights = np.array([0.5, 0.5])
        
        # Symmetric returns - no drift
        monthly_returns = np.array([
            [0.01, 0.01],
            [0.01, 0.01],
            [-0.01, -0.01],
        ])
        
        initial_investment = 100000
        rebalance_threshold = 0.05
        num_months = 3
        contrib = 1000
        
        standard_values, _, _ = _simulate_portfolio_numba(
            target_weights, monthly_returns, contrib,
            initial_investment, rebalance_threshold, num_months,
            asymmetric_rebalancing=False
        )
        
        asymmetric_values, _, _ = _simulate_portfolio_numba(
            target_weights, monthly_returns, contrib,
            initial_investment, rebalance_threshold, num_months,
            asymmetric_rebalancing=True
        )
        
        # With no drift, both should produce very similar results
        assert np.allclose(standard_values, asymmetric_values, rtol=0.01), \
            "Both strategies should produce similar results with no drift"


class TestAsymmetricRebalancingEdgeCases:
    """Test edge cases for asymmetric rebalancing."""
    
    def test_handles_zero_portfolio_value(self):
        """Test that asymmetric rebalancing handles zero portfolio value."""
        target_weights = np.array([0.5, 0.5])
        holdings_value = np.array([0.0, 0.0])
        contrib = 1000.0
        
        # Should not crash
        allocation = _allocate_contribution_asymmetric(holdings_value, target_weights, contrib)
        
        # Should allocate to target weights
        assert allocation[0] == pytest.approx(500.0, abs=1.0)
        assert allocation[1] == pytest.approx(500.0, abs=1.0)
    
    def test_handles_single_asset_portfolio(self):
        """Test asymmetric rebalancing with single asset."""
        target_weights = np.array([1.0])
        holdings_value = np.array([100000.0])
        
        needs_rebalance = _check_asymmetric_rebalance_needed(holdings_value, target_weights)
        
        # Single asset can't exceed its band (it's always 100% of portfolio)
        # High band is 1.0 * 2 = 2.0, current weight is 1.0
        assert not needs_rebalance, "Single asset should not trigger rebalance"
    
    def test_three_asset_portfolio(self):
        """Test asymmetric rebalancing with three assets."""
        target_weights = np.array([0.5, 0.3, 0.2])
        
        # One asset way over, one under, one balanced
        # High bands: 100%, 60%, 40%
        holdings_value = np.array([110000.0, 5000.0, 20000.0])  # 81.5%, 3.7%, 14.8%
        
        needs_rebalance = _check_asymmetric_rebalance_needed(holdings_value, target_weights)
        
        # First asset at 81.5% is below its high band of 100%
        # So should NOT trigger rebalance
        assert not needs_rebalance, "Should not rebalance when within high bands"
        
        # Now make first asset exceed band
        holdings_value = np.array([150000.0, 5000.0, 5000.0])  # 93.75%, 3.1%, 3.1%
        needs_rebalance = _check_asymmetric_rebalance_needed(holdings_value, target_weights)
        
        # First asset at 93.75% is below 100% high band... wait that's wrong
        # Let me recalculate: if target is 50%, high band is 50%*2 = 100%
        # Current weight is 150k/160k = 93.75%
        # 93.75% < 100%, so still not over
        # Let me make it clearly over
        holdings_value = np.array([250000.0, 5000.0, 5000.0])  # 96.2%, 1.9%, 1.9%
        needs_rebalance = _check_asymmetric_rebalance_needed(holdings_value, target_weights)
        # 96.2% < 100%, still not over!
        
        # To exceed 100% high band, asset needs to be >100% of portfolio, which is impossible
        # unless other assets are negative. Let me rethink...
        
        # Actually the high band is "target weight * 2"
        # So if target is 50% of portfolio, high band is 100% of portfolio
        # An asset at 90% of portfolio is still within band
        # To exceed, it would need to be >100% which only happens with negative assets
        
        # Let me use a smaller target to make this testable
        # Actually, I think the issue is I need to test with the actual constraint
        # If target is 20%, high band is 40%, which is achievable
        holdings_value = np.array([30000.0, 30000.0, 50000.0])  # 27%, 27%, 45%
        needs_rebalance = _check_asymmetric_rebalance_needed(holdings_value, target_weights)
        # Third asset: target 20%, high band 40%, current 45%
        # 45% > 40%, should trigger!
        assert needs_rebalance, "Should rebalance when asset exceeds high band"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

