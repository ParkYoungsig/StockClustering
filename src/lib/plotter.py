from pathlib import Path
from typing import Optional, List, Tuple
from logging_config import logger

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter

import pandas as pd
import numpy as np
import mplfinance as mpf

  

class Plotter:

    def __init__(self, output_dir: str = 'output'):
        """ output_dir: Directory for output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Color scheme
        self.colors = {
            'up': '#26a69a',        # Teal
            'down': '#ef5350',      # Red
            'volume': '#6495ED',    # Cornflower blue
            'ma5': '#FF6B6B',       # Light red
            'ma20': '#4ECDC4',      # Turquoise
            'ma50': '#FFE66D',      # Yellow
            'positive': '#2ecc71',  # Green
            'negative': '#e74c3c',  # Red
            'neutral': '#3498db'    # Blue
        }
        
        logger.info(f"output-dir: {self.output_dir}")
    
    def plot_price_chart(
        self,
        df: pd.DataFrame,
        ticker: str,
        ma_periods: List[int] = [5, 20, 50],
        save_path: Optional[str] = None
    ) -> str:
        """
            df: OHLCV dataframe
            ticker: Stock ticker
            ma_periods: Moving average periods
            save_path: Custom save path (optional)
        """
        
        # Calculate moving averages
        for period in ma_periods:
            df[f'MA{period}'] = df['Close'].rolling(window=period).mean()
        
        # Prepare additional plots
        apds = []
        
        # Add moving averages
        ma_colors = [self.colors['ma5'], self.colors['ma20'], self.colors['ma50']]
        for i, period in enumerate(ma_periods[:3]):  # Limit to 3 MAs
            if f'MA{period}' in df.columns:
                apds.append(
                    mpf.make_addplot(
                        df[f'MA{period}'],
                        color=ma_colors[i],
                        width=1.5,
                        alpha=0.8,
                        label=f'MA{period}'
                    )
                )
        
        # Custom style
        mc = mpf.make_marketcolors(
            up=self.colors['up'],
            down=self.colors['down'],
            volume=self.colors['volume'],
            edge='inherit',
            wick='inherit'
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='-',
            gridcolor='#E0E0E0',
            gridaxis='both',
            facecolor='white',
            figcolor='white',
            y_on_right=False
        )
        
        # Create figure
        if save_path is None:
            save_path = self.output_dir / f"{ticker}_price_chart.png"
        else:
            save_path = Path(save_path)
        
        # Plot
        fig, axes = mpf.plot(
            df,
            type='candle',
            style=s,
            volume=True,
            addplot=apds,
            title=f'{ticker} - Price & Volume',
            ylabel='Price',
            ylabel_lower='Volume',
            figsize=(14, 8),
            returnfig=True,
            datetime_format='%Y-%m-%d',
            xrotation=45
        )
        
        # Add legend for MAs
        if apds:
            axes[0].legend([f'MA{p}' for p in ma_periods[:len(apds)]], 
                          loc='upper left')
        
        # Save figure
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Price chart saved to {save_path}")
        return str(save_path)
    
    def plot_risk_metrics(
        self,
        metrics_df: pd.DataFrame,
        ticker: str,
        benchmark: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create comprehensive risk metrics dashboard
        
        input:
            metrics_df: DataFrame with risk metrics
            ticker: Stock ticker
            benchmark: Benchmark ticker (optional)
            save_path: Custom save path (optional)
            
        return:
            str: Path to saved chart
        """
        logger.info(f"Creating risk metrics chart for {ticker}")
        
        # Determine available metrics
        available_metrics = metrics_df.columns.tolist()
        
        # Define metric groups
        ratio_metrics = ['sharpe', 'sortino', 'calmar', 'omega', 'information_ratio']
        ratio_metrics = [m for m in ratio_metrics if m in available_metrics]
        
        market_metrics = ['alpha', 'beta']
        market_metrics = [m for m in market_metrics if m in available_metrics]
        
        performance_metrics = ['win_rate', 'profit_factor', 'kelly']
        performance_metrics = [m for m in performance_metrics if m in available_metrics]
        
        # Create figure with subplots
        n_plots = 0
        if ratio_metrics:
            n_plots += 1
        if market_metrics:
            n_plots += 1
        if performance_metrics:
            n_plots += 1
        
        if n_plots == 0:
            logger.warning("No metrics available to plot")
            return ""
        
        fig = plt.figure(figsize=(14, 4 * n_plots))
        gs = gridspec.GridSpec(n_plots, 1, hspace=0.3)
        
        plot_idx = 0
        
        # Plot ratio metrics
        if ratio_metrics:
            ax = fig.add_subplot(gs[plot_idx])
            self._plot_metrics_group(
                ax, metrics_df, ratio_metrics,
                f'{ticker} - Risk-Adjusted Return Metrics',
                'Ratio'
            )
            plot_idx += 1
        
        # Plot market metrics
        if market_metrics:
            ax = fig.add_subplot(gs[plot_idx])
            self._plot_metrics_group(
                ax, metrics_df, market_metrics,
                f'{ticker} vs {benchmark} - Market Metrics' if benchmark else f'{ticker} - Market Metrics',
                'Value'
            )
            plot_idx += 1
        
        # Plot performance metrics
        if performance_metrics:
            ax = fig.add_subplot(gs[plot_idx])
            self._plot_metrics_group(
                ax, metrics_df, performance_metrics,
                f'{ticker} - Performance Metrics',
                'Value'
            )
            plot_idx += 1
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / f"{ticker}_risk_metrics.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Risk metrics chart saved to {save_path}")
        return str(save_path)
    
    def _plot_metrics_group(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        metrics: List[str],
        title: str,
        ylabel: str
    ):
        
        for metric in metrics:
            if metric in df.columns:
                # Clean data
                data = df[metric].dropna()
                if len(data) > 0:
                    ax.plot(
                        data.index,
                        data.values,
                        label=metric.replace('_', ' ').title(),
                        linewidth=1.5,
                        alpha=0.8
                    )
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel('Date', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add zero line for reference
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    def plot_combined_dashboard(
        self,
        price_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        ticker: str,
        benchmark: Optional[str] = None,
        ma_periods: List[int] = [5, 20, 50],
        save_path: Optional[str] = None
    ) -> str:
        """
            price_df: OHLCV dataframe
            metrics_df: Risk metrics dataframe
            ticker: Stock ticker
            benchmark: Benchmark ticker (optional)
            ma_periods: Moving average periods
            save_path: Custom save path (optional)
        """
        
        # Create individual charts first
        price_path = self.plot_price_chart(price_df, ticker, ma_periods)
        metrics_path = self.plot_risk_metrics(metrics_df, ticker, benchmark)
        
        # For now, return both paths
        # In future, could combine into single figure
        logger.info(f"Dashboard created: price={price_path}, metrics={metrics_path}")
        
        return price_path
    
    def plot_drawdown(
        self,
        returns: pd.Series,
        ticker: str,
        save_path: Optional[str] = None
    ) -> str:
        """
            return: Returns series
            ticker: Stock ticker
            save_path: Custom save path (optional)
        """

        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cum_returns - running_max) / running_max
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Plot cumulative returns
        ax1.plot(cum_returns.index, cum_returns.values, 
                color=self.colors['positive'], linewidth=2, label='Cumulative Returns')
        ax1.plot(running_max.index, running_max.values, 
                color=self.colors['neutral'], linestyle='--', 
                linewidth=1.5, alpha=0.7, label='Running Maximum')
        ax1.set_title(f'{ticker} - Cumulative Returns', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=10)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdown
        ax2.fill_between(drawdown.index, 0, drawdown.values, 
                        color=self.colors['negative'], alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, 
                color=self.colors['negative'], linewidth=1.5)
        ax2.set_title(f'{ticker} - Drawdown', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown', fontsize=10)
        ax2.set_xlabel('Date', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Format as percentage
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / f"{ticker}_drawdown.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Drawdown chart saved to {save_path}")
        return str(save_path)
    
    def plot_returns_distribution(
        self,
        returns: pd.Series,
        ticker: str,
        save_path: Optional[str] = None
    ) -> str:
        """
            return: Returns series
            ticker: Stock ticker
            save_path: Custom save path (optional)
        """
                
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        n, bins, patches = ax.hist(
            returns.dropna(),
            bins=50,
            alpha=0.7,
            color=self.colors['neutral'],
            edgecolor='black'
        )
        
        # Color bars based on positive/negative
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor(self.colors['negative'])
            else:
                patch.set_facecolor(self.colors['positive'])
        
        # Add statistics
        mean_return = returns.mean()
        std_return = returns.std()
        
        ax.axvline(mean_return, color='black', linestyle='--', 
                  linewidth=2, label=f'Mean: {mean_return:.4f}')
        ax.axvline(0, color='gray', linestyle='-', 
                  linewidth=1, alpha=0.5)
        
        # Add text box with stats
        stats_text = f'Mean: {mean_return:.4f}\nStd: {std_return:.4f}\nSkew: {returns.skew():.4f}\nKurt: {returns.kurtosis():.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
        ax.set_title(f'{ticker} - Returns Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Daily Returns', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / f"{ticker}_returns_dist.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Returns distribution saved to {save_path}")
        return str(save_path)


if __name__ == "__main__":
    # Test the plotter
    import yfinance as yf
    
    # Fetch sample data
    ticker = 'AAPL'
    df = yf.download(ticker, start='2020-01-01', end='2025-12-12', progress=False)
    
    # Create plotter
    plotter = Plotter()
    
    # Test price chart
    price_path = plotter.plot_price_chart(df, ticker)
    print(f"Price chart saved to: {price_path}")
    
    # Test returns distribution
    returns = df['Close'].pct_change().dropna()
    dist_path = plotter.plot_returns_distribution(returns, ticker)
    print(f"Distribution chart saved to: {dist_path}")
    
    # Test drawdown
    dd_path = plotter.plot_drawdown(returns, ticker)
    print(f"Drawdown chart saved to: {dd_path}")