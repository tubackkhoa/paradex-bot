"""
Paradex Volume Generator Bot - Fully Configurable via .env
Target: Customizable volume in configurable timeframe
Strategy: Ultra-tight spread with all parameters in .env file
Features: Zero-fee trading, High-frequency execution, Real-time monitoring
"""
import asyncio
import os
import signal
import threading
from datetime import datetime, timedelta
from decimal import Decimal
from dotenv import load_dotenv
from paradex_py import Paradex
from paradex_py.environment import Environment
from paradex_py.common.order import Order, OrderType, OrderSide
import time

load_dotenv()

class ParadexVolumeBot:
    def __init__(self):
        # ===== API Configuration =====
        self.l2_private_key = os.getenv('L2_PRIVATE_KEY')
        self.l1_private_key = os.getenv('L1_PRIVATE_KEY')
        self.l1_address = os.getenv('L1_ADDRESS')
        self.environment = os.getenv('ENVIRONMENT', 'TESTNET').upper()
        
        # ===== Market & Trading Settings =====
        self.market = os.getenv('MARKET', 'BTC-USD-PERP')
        self.leverage = int(os.getenv('LEVERAGE', 10))
        self.investment = float(os.getenv('INVESTMENT_USDC', 10))
        
        # ===== Volume Target Settings =====
        self.target_volume = float(os.getenv('TARGET_VOLUME', 100000))
        self.max_loss = float(os.getenv('MAX_LOSS', 10))
        self.target_hours = int(os.getenv('TARGET_HOURS', 24))
        
        # ===== Strategy Parameters =====
        self.spread_bps = float(os.getenv('SPREAD_BPS', 2))  # 0.02% default
        self.orders_per_side = int(os.getenv('ORDERS_PER_SIDE', 10))
        self.order_size_percent = float(os.getenv('ORDER_SIZE_PERCENT', 0.1))  # 10% per order
        self.refresh_interval = float(os.getenv('REFRESH_INTERVAL', 2.0))
        
        # ===== Rate Limit Protection =====
        self.delay_between_orders = float(os.getenv('DELAY_BETWEEN_ORDERS', 0.05))
        self.delay_after_cancel = float(os.getenv('DELAY_AFTER_CANCEL', 0.3))
        self.status_interval = int(os.getenv('STATUS_INTERVAL', 30))
        self.max_orders_to_place = int(os.getenv('MAX_ORDERS_TO_PLACE', 10))
        
        # ===== Advanced Settings =====
        self.use_post_only = os.getenv('USE_POST_ONLY', 'true').lower() == 'true'
        self.trading_fee_percent = float(os.getenv('TRADING_FEE_PERCENT', 0.0))
        
        # Calculate derived metrics
        self.hourly_target = self.target_volume / self.target_hours
        self.trades_needed = int(self.target_volume / 10)
        self.avg_trade_size = self.target_volume / self.trades_needed
        
        # Paradex client
        self.paradex = None
        self.client_order_id = int(time.time() * 1000)
        
        # Tracking
        self.running = True
        self.active_orders = {}
        self.total_volume = 0.0
        self.total_trades = 0
        self.total_fees = 0.0
        self.session_start = None
        self.last_fill_time = time.time()
        
        # Hourly tracking
        self.current_hour_volume = 0.0
        self.current_hour_trades = 0
        self.hour_start = None
        self.hourly_stats = []
        
        # Market info cache
        self.market_info = None
        self.tick_size = None
        self.step_size = None

    async def init(self):
        """Initialize Paradex client"""
        from paradex_py.environment import TESTNET, PROD
        
        env = TESTNET if self.environment == 'TESTNET' else PROD
        
        try:
            self.paradex = Paradex(
                env=env,
                l1_address=self.l1_address,
                l2_private_key=self.l2_private_key,
                l1_private_key=self.l1_private_key
            )
            
            # Check if account is onboarded
            print(f"🔐 Checking account status...")
            try:
                account_info = self.paradex.api_client.fetch_account_profile()
                print(f"✅ Account onboarded successfully")
            except Exception as e:
                print(f"⚠️  Account not onboarded yet. Attempting onboarding...")
                # The SDK should handle onboarding automatically
                # Just try to fetch again after a delay
                await asyncio.sleep(2)
                try:
                    account_info = self.paradex.api_client.fetch_account_profile()
                    print(f"✅ Account onboarded successfully")
                except:
                    print(f"❌ Onboarding failed. Please onboard manually at https://app.paradex.trade")
                    raise Exception("Account not onboarded")
        
        except Exception as e:
            print(f"❌ Failed to initialize Paradex client: {e}")
            raise
        
        self.session_start = datetime.now()
        self.hour_start = datetime.now()
        
        # Fetch market info
        await self.fetch_market_info()
        
        print(f"{'='*75}")
        print(f"🚀 PARADEX VOLUME GENERATOR - FULLY CONFIGURABLE")
        print(f"{'='*75}")
        print(f"Environment: {self.environment}")
        print(f"Market: {self.market}")
        print(f"Account: {self.l1_address[:10]}...{self.l1_address[-8:]}")
        print(f"Investment: ${self.investment:.2f} (Leverage: {self.leverage}x)")
        print(f"Effective Capital: ${self.investment * self.leverage:.2f}")
        print(f"\n🎯 TARGETS:")
        print(f"   Volume Goal: ${self.target_volume:,.0f} in {self.target_hours}h")
        print(f"   Hourly Goal: ${self.hourly_target:,.0f}")
        print(f"   Max Loss: ${self.max_loss:.2f}")
        print(f"\n⚙️  STRATEGY CONFIG:")
        print(f"   Spread: {self.spread_bps/100:.3f}% ({self.spread_bps} bps)")
        print(f"   Orders: {self.orders_per_side*2} total ({self.orders_per_side} each side)")
        print(f"   Order Size: {self.order_size_percent*100:.1f}% of capital")
        print(f"   Refresh: Every {self.refresh_interval}s")
        print(f"\n🛡️  RATE LIMIT PROTECTION:")
        print(f"   Delay Between Orders: {self.delay_between_orders}s")
        print(f"   Delay After Cancel: {self.delay_after_cancel}s")
        print(f"   Max Orders/Cycle: {self.max_orders_to_place} per side")
        print(f"   Status Updates: Every {self.status_interval}s")
        print(f"   Rate Limit: 800 req/s, 17250 req/min")
        print(f"\n💡 PROJECTIONS:")
        print(f"   Est. Trades Needed: ~{self.trades_needed:,}")
        print(f"   Avg Trade Size: ${self.avg_trade_size:.2f}")
        print(f"   Trading Fee: {self.trading_fee_percent}% {'🎉 ZERO FEE!' if self.trading_fee_percent == 0 else ''}")
        print(f"   Order Type: {'POST_ONLY' if self.use_post_only else 'LIMIT'}")
        print(f"{'='*75}\n")

    async def fetch_market_info(self):
        """Fetch market configuration"""
        try:
            print(f"📊 Fetching market info for {self.market}...")
            markets = self.paradex.api_client.fetch_markets()
            
            # Filter perpetual markets only
            perp_markets = []
            for market in markets.get('results', []):
                symbol = market.get('symbol', '')
                # Perpetuals end with -PERP
                if '-PERP' in symbol or market.get('type') == 'PERP':
                    perp_markets.append(market)
                    
                    if symbol == self.market:
                        self.market_info = market
                        
                        # Parse tick_size and step_size from market config
                        # Try different field names
                        self.tick_size = float(
                            market.get('tick_size') or 
                            market.get('price_tick_size') or 
                            market.get('min_price_increment') or
                            0.1
                        )
                        self.step_size = float(
                            market.get('step_size') or 
                            market.get('size_increment') or 
                            market.get('min_size') or
                            0.001
                        )
                        
                        print(f"✅ Market Info Loaded:")
                        print(f"   Symbol: {symbol}")
                        print(f"   Type: {market.get('type', 'PERP')}")
                        print(f"   Tick Size: {self.tick_size}")
                        print(f"   Step Size: {self.step_size}")
                        print(f"   Status: {market.get('status', 'UNKNOWN')}")
                        return
            
            # Market not found in perpetuals
            print(f"⚠️  Market {self.market} not found")
            print(f"\n📋 Available Perpetual Markets ({len(perp_markets)} total):")
            
            # Show popular markets
            popular = ['BTC-USD-PERP', 'ETH-USD-PERP', 'SOL-USD-PERP', 'DOGE-USD-PERP', 'AVAX-USD-PERP']
            shown = []
            for p in popular:
                if any(m.get('symbol') == p for m in perp_markets):
                    shown.append(p)
            
            # Add first 10 perpetuals if not enough popular ones
            for m in perp_markets[:15]:
                s = m.get('symbol', '')
                if s not in shown:
                    shown.append(s)
                if len(shown) >= 15:
                    break
            
            print(f"   {', '.join(shown)}")
            print(f"\n💡 Update .env file: MARKET=BTC-USD-PERP")
            print(f"   Using defaults: tick_size=0.1, step_size=0.001")
            
            self.tick_size = 0.1
            self.step_size = 0.001
                    
        except Exception as e:
            print(f"⚠️  Error fetching market info: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n   Using defaults: tick_size=0.1, step_size=0.001")
            self.tick_size = 0.1
            self.step_size = 0.001

    async def get_orderbook(self):
        """Get current orderbook"""
        try:
            bbo = self.paradex.api_client.fetch_bbo(self.market)
            
            if bbo:
                # Paradex BBO format: 'bid' and 'ask' (not 'best_bid'/'best_ask')
                best_bid = float(bbo.get('bid', bbo.get('best_bid', 0)))
                best_ask = float(bbo.get('ask', bbo.get('best_ask', 0)))
                
                if best_bid > 0 and best_ask > 0:
                    mid_price = (best_bid + best_ask) / 2
                    spread_pct = ((best_ask - best_bid) / mid_price) * 100
                    
                    return {
                        'best_bid': best_bid,
                        'best_ask': best_ask,
                        'mid_price': mid_price,
                        'spread_pct': spread_pct
                    }
            return None
        except Exception as e:
            # Silent fail to avoid spam, will retry next cycle
            return None

    def round_price(self, price):
        """Round price to tick size"""
        if self.tick_size:
            # Calculate decimal places from tick_size
            import math
            if self.tick_size >= 1:
                # tick_size = 1, 10, etc - round to integer
                decimals = 0
            else:
                # tick_size = 0.1, 0.01, 0.00001, etc
                decimals = abs(int(math.floor(math.log10(self.tick_size))))
            
            return round(price, decimals)
        return round(price, 2)

    def round_size(self, size):
        """Round size to step size"""
        if self.step_size:
            rounded = round(size / self.step_size) * self.step_size
            # For assets that require integer, round to integer
            if self.step_size >= 1:
                return int(round(size))
            # If rounded to 0 but size > 0, use minimum step
            if rounded == 0 and size > 0:
                return self.step_size
            return rounded
        return round(size, 6)

    async def calculate_order_levels(self, orderbook):
        """Calculate order levels with configurable spread"""
        mid_price = orderbook['mid_price']
        best_bid = orderbook['best_bid']
        best_ask = orderbook['best_ask']
        
        spread = mid_price * (self.spread_bps / 10000)
        
        buy_levels = []
        sell_levels = []
        
        # Buy orders
        for i in range(self.orders_per_side):
            price = self.round_price(best_bid - (spread * i * 0.4))
            buy_levels.append(price)
        
        # Sell orders
        for i in range(self.orders_per_side):
            price = self.round_price(best_ask + (spread * i * 0.4))
            sell_levels.append(price)
        
        return buy_levels, sell_levels

    async def place_order(self, price, side, size):
        """Place single order"""
        try:
            self.client_order_id += 1
            
            # Convert string side to OrderSide enum
            order_side = OrderSide.Buy if side == "BUY" else OrderSide.Sell
            
            # Convert to Decimal for Paradex SDK
            order = Order(
                market=self.market,
                order_type=OrderType.Limit,
                order_side=order_side,
                size=Decimal(str(self.round_size(size))),  # Must be Decimal
                limit_price=Decimal(str(price)),  # Must be Decimal
                client_id=str(self.client_order_id),
                instruction='POST_ONLY' if self.use_post_only else 'GTC',
                reduce_only=False
            )
            
            # Debug: print order details for first few attempts
            if self.client_order_id <= 3:
                print(f"   📝 Submitting order #{self.client_order_id}:")
                print(f"      Side: {side}")
                print(f"      Size: {order.size}")
                print(f"      Price: {order.limit_price}")
                print(f"      Instruction: {order.instruction}")
            
            result = self.paradex.api_client.submit_order(order)
            
            # Debug: print full response for first few attempts
            if self.client_order_id <= 3:
                print(f"      Result type: {type(result)}")
                print(f"      Result value: {result}")
                if result:
                    print(f"      Result keys: {result.keys() if hasattr(result, 'keys') else 'N/A'}")
            
            if result and isinstance(result, dict) and 'id' in result:
                self.active_orders[result['id']] = {
                    'client_id': self.client_order_id,
                    'price': price,
                    'side': side,
                    'size': size,
                    'timestamp': time.time()
                }
                if self.client_order_id <= 3:
                    print(f"      ✅ SUCCESS - Order ID: {result['id']}")
                return True
            else:
                # Print failures
                if self.client_order_id <= 10:
                    print(f"   ❌ Order rejected: {side} @ ${price:.5f}")
                    print(f"      Result was: {result}")
                    print(f"      Has 'id'?: {'id' in result if isinstance(result, dict) else 'Not a dict'}")
                return False
                
        except Exception as e:
            # Print all errors with full traceback
            print(f"   ❌ EXCEPTION placing order: {side} @ ${price:.5f}")
            print(f"      Error type: {type(e).__name__}")
            print(f"      Error message: {str(e)}")
            if self.client_order_id <= 3:
                import traceback
                traceback.print_exc()
            return False

    async def cancel_all_orders(self):
        """Cancel all active orders"""
        try:
            self.paradex.api_client.cancel_all_orders({'market': self.market})
            self.active_orders.clear()
        except Exception as e:
            pass

    async def refresh_orders(self):
        """Main order refresh loop"""
        print(f"🔄 Starting order refresh ({self.refresh_interval}s cycles)...\n")
        
        cycle = 0
        last_status_time = time.time()
        
        while self.running:
            try:
                cycle += 1
                cycle_start = time.time()
                
                # Get orderbook
                orderbook = await self.get_orderbook()
                if not orderbook:
                    print(f"   ⚠️  Cycle {cycle}: No orderbook data, retrying...")
                    await asyncio.sleep(self.refresh_interval)
                    continue
                
                # Print orderbook info for first few cycles
                if cycle <= 3:
                    print(f"\n📊 Cycle {cycle} - Orderbook:")
                    print(f"   Best Bid: ${orderbook['best_bid']:,.2f}")
                    print(f"   Best Ask: ${orderbook['best_ask']:,.2f}")
                    print(f"   Mid Price: ${orderbook['mid_price']:,.2f}")
                    print(f"   Spread: {orderbook['spread_pct']:.3f}%")
                
                # Cancel existing orders
                await self.cancel_all_orders()
                await asyncio.sleep(self.delay_after_cancel)
                
                # Calculate levels
                buy_levels, sell_levels = await self.calculate_order_levels(orderbook)
                
                # Calculate order size
                coin_size = (self.investment * self.leverage * self.order_size_percent) / orderbook['mid_price']
                
                if cycle <= 3:
                    print(f"   Order size: {coin_size:.6f} {self.market.split('-')[0]}")
                    print(f"   Rounded size: {self.round_size(coin_size):.6f}")
                    print(f"   Placing {self.max_orders_to_place} buy + {self.max_orders_to_place} sell orders...")
                
                # Place buy orders
                placed_buy = 0
                for i, price in enumerate(buy_levels[:self.max_orders_to_place]):
                    success = await self.place_order(price, "BUY", coin_size)
                    if success:
                        placed_buy += 1
                        if cycle <= 3 and i < 3:
                            print(f"   ✅ BUY @ ${price:,.5f}")
                    elif cycle <= 3 and i < 3:
                        print(f"   ⚠️  BUY @ ${price:,.5f} - Failed (no error captured)")
                    await asyncio.sleep(self.delay_between_orders)
                
                # Place sell orders
                placed_sell = 0
                for i, price in enumerate(sell_levels[:self.max_orders_to_place]):
                    success = await self.place_order(price, "SELL", coin_size)
                    if success:
                        placed_sell += 1
                        if cycle <= 3 and i < 3:
                            print(f"   ✅ SELL @ ${price:,.5f}")
                    elif cycle <= 3 and i < 3:
                        print(f"   ⚠️  SELL @ ${price:,.5f} - Failed (no error captured)")
                    await asyncio.sleep(self.delay_between_orders)
                
                if cycle <= 3:
                    print(f"   Summary: {placed_buy} buy + {placed_sell} sell orders placed\n")
                
                # Estimate fills
                estimated_fills = max(0, (self.max_orders_to_place - placed_buy) + (self.max_orders_to_place - placed_sell))
                
                if estimated_fills > 0:
                    fill_volume = estimated_fills * coin_size * orderbook['mid_price']
                    self.total_volume += fill_volume
                    self.current_hour_volume += fill_volume
                    self.total_trades += estimated_fills
                    self.current_hour_trades += estimated_fills
                    
                    # Calculate fees
                    trade_fees = fill_volume * (self.trading_fee_percent / 100)
                    self.total_fees += trade_fees
                
                # Print status
                if time.time() - last_status_time >= self.status_interval:
                    await self.print_status(orderbook, placed_buy, placed_sell)
                    last_status_time = time.time()
                
                # Hour rollover
                if (datetime.now() - self.hour_start).total_seconds() >= 3600:
                    self.hourly_stats.append({
                        'volume': self.current_hour_volume,
                        'trades': self.current_hour_trades
                    })
                    print(f"\n⏰ HOUR {len(self.hourly_stats)} COMPLETE:")
                    print(f"   Volume: ${self.current_hour_volume:,.0f}")
                    print(f"   Trades: {self.current_hour_trades:,}")
                    print(f"   Target: ${self.hourly_target:,.0f}")
                    print(f"   Status: {'✅ ON TRACK' if self.current_hour_volume >= self.hourly_target * 0.8 else '⚠️  BEHIND'}\n")
                    
                    self.current_hour_volume = 0.0
                    self.current_hour_trades = 0
                    self.hour_start = datetime.now()
                
                # Safety check
                if self.trading_fee_percent > 0 and self.total_fees >= self.max_loss:
                    print(f"\n🛑 MAX LOSS REACHED: ${self.total_fees:.2f}")
                    self.running = False
                    break
                
                # Sleep
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.refresh_interval - cycle_time)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                print(f"⚠️  Cycle error: {e}")
                await asyncio.sleep(self.refresh_interval)

    async def print_status(self, orderbook, placed_buy, placed_sell):
        """Print status update"""
        runtime = datetime.now() - self.session_start
        hours_run = runtime.total_seconds() / 3600
        
        # Get actual fills from API
        try:
            fills = self.paradex.api_client.fetch_fills({'market': self.market})
            if fills and 'results' in fills:
                # Count only fills from this session
                session_fills = [f for f in fills['results'] 
                               if int(f.get('created_at', 0)) >= int(self.session_start.timestamp() * 1000)]
                
                # Calculate real volume
                real_volume = sum(float(f.get('size', 0)) * float(f.get('price', 0)) 
                                for f in session_fills)
                real_trades = len(session_fills)
                
                # Update tracking with real data
                self.total_volume = real_volume
                self.total_trades = real_trades
        except:
            # If can't fetch fills, show warning
            real_volume = 0
            real_trades = 0
        
        volume_rate = self.total_volume / max(hours_run, 0.01)
        trade_rate = self.total_trades / max(hours_run, 0.01)
        projected = volume_rate * self.target_hours
        progress_pct = (self.total_volume / self.target_volume) * 100
        
        time_remaining = timedelta(hours=self.target_hours) - runtime
        hours_left = time_remaining.total_seconds() / 3600
        volume_left = self.target_volume - self.total_volume
        required_rate = volume_left / max(hours_left, 0.01) if hours_left > 0 else 0
        
        print(f"{'='*75}")
        print(f"⏱️  {str(runtime).split('.')[0]} elapsed | {max(0, hours_left):.1f}h left | Price: ${orderbook['mid_price']:,.2f}")
        print(f"📊 Orders: {placed_buy} BUY + {placed_sell} SELL | Spread: {orderbook['spread_pct']:.3f}%")
        print(f"\n💰 VOLUME (REAL from API):")
        print(f"   Current: ${self.total_volume:,.0f} / ${self.target_volume:,.0f} ({progress_pct:.1f}%)")
        print(f"   This Hour: ${self.current_hour_volume:,.0f} / ${self.hourly_target:,.0f}")
        print(f"   Trades: {self.total_trades:,} ({trade_rate:.0f}/hour)")
        print(f"\n📈 PERFORMANCE:")
        print(f"   Current Rate: ${volume_rate:,.0f}/hour")
        print(f"   {self.target_hours}h Projection: ${projected:,.0f}")
        print(f"   Required Rate: ${required_rate:,.0f}/hour")
        print(f"   Status: {'✅ ON TRACK' if volume_rate >= required_rate * 0.9 else '⚠️  SPEED UP'}")
        
        if self.trading_fee_percent == 0:
            print(f"\n💸 COSTS:")
            print(f"   🎉 ZERO FEES - Free trading!")
            print(f"   Loss (spread): ${self.total_fees:.2f}")
        else:
            print(f"\n💸 COSTS:")
            print(f"   Fees: ${self.total_fees:.2f} / ${self.max_loss:.2f}")
            print(f"   Budget Left: ${self.max_loss - self.total_fees:.2f}")
        print(f"{'='*75}\n")

    def stop_bot(self, signum=None, frame=None):
        """Stop bot gracefully"""
        print("\nSTOPPING BOT...")
        self.running = False

    async def run(self):
        """Main execution"""
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self.stop_bot)
        
        try:
            await self.init()
            await self.refresh_orders()
            
        except KeyboardInterrupt:
            self.stop_bot()
        except Exception as e:
            print(f"❌ Fatal Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n🧹 Cleaning up...")
            
            # Only cancel if paradex client is initialized
            if self.paradex:
                await self.cancel_all_orders()
            
            # Only calculate runtime if session started
            if self.session_start:
                runtime = datetime.now() - self.session_start
                hours_run = runtime.total_seconds() / 3600
                
                print(f"\n{'='*75}")
                print(f"📊 FINAL REPORT - PARADEX")
                print(f"{'='*75}")
                print(f"Runtime: {str(runtime).split('.')[0]} ({hours_run:.2f} hours)")
                print(f"\n💰 VOLUME:")
                print(f"   Total: ${self.total_volume:,.2f}")
                print(f"   Target: ${self.target_volume:,.0f}")
                print(f"   Achievement: {(self.total_volume/self.target_volume)*100:.1f}%")
                print(f"   Hourly Avg: ${self.total_volume/max(hours_run,0.01):,.0f}/hour")
                print(f"\n📈 TRADES:")
                print(f"   Total: {self.total_trades:,}")
                print(f"   Avg/Hour: {self.total_trades/max(hours_run,0.01):.0f}")
                print(f"   Avg Size: ${self.total_volume/max(self.total_trades,1):.2f}")
                
                if self.trading_fee_percent == 0:
                    print(f"\n💸 COSTS:")
                    print(f"   🎉 ZERO FEES!")
                    print(f"   Loss: ${self.total_fees:.2f} (spread only)")
                else:
                    print(f"\n💸 COSTS:")
                    print(f"   Fees: ${self.total_fees:.2f}")
                    print(f"   Budget: ${self.max_loss:.2f}")
                    print(f"   Used: {(self.total_fees/self.max_loss)*100:.1f}%")
                
                print(f"{'='*75}\n")
            
            print("👋 Bot stopped\n")

def start():
    """Entry point to start the bot"""
    bot = ParadexVolumeBot()
    asyncio.run(bot.run())

if __name__ == "__main__":
    start()    