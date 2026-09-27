"""
连续双边拍卖(CDA)市场撮合引擎
实现限价订单簿和交易撮合逻辑
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
from collections import deque
import heapq
import numpy as np


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """订单类型"""
    LIMIT = "limit"  # 限价单
    MARKET = "market"  # 市价单


@dataclass
class Order:
    """订单类"""
    order_id: str
    agent_id: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # 市价单为None
    timestamp: int = 0
    
    def __post_init__(self):
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("限价单必须指定价格")
    
    def __lt__(self, other):
        """用于优先队列排序"""
        if self.side != other.side:
            return self.side.value < other.side.value
        # 买单：价格降序；卖单：价格升序
        if self.side == OrderSide.BUY:
            if self.price != other.price:
                return self.price > other.price
        else:
            if self.price != other.price:
                return self.price < other.price
        return self.timestamp < other.timestamp


@dataclass
class Trade:
    """成交记录"""
    trade_id: str
    buyer_id: str
    seller_id: str
    price: float
    quantity: float
    timestamp: int
    buy_order_id: str
    sell_order_id: str


@dataclass
class MarketSnapshot:
    """市场快照"""
    timestamp: int
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    mid_price: Optional[float] = None
    spread: Optional[float] = None
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    last_price: Optional[float] = None
    volume_24h: float = 0.0


class OrderBook:
    """限价订单簿"""
    
    def __init__(self, tick_size: float = 0.1):
        self.tick_size = tick_size
        
        # 买单簿：价格降序 (使用最大堆，存储负价格)
        self.bids: List[Tuple[float, int, Order]] = []  # (-price, timestamp, order)
        # 卖单簿：价格升序
        self.asks: List[Tuple[float, int, Order]] = []  # (price, timestamp, order)
        
        # 订单映射，便于取消
        self.orders: Dict[str, Order] = {}
        
        # 订单计数器
        self._order_counter = 0
        self._trade_counter = 0
        
        # 历史成交
        self.trades: List[Trade] = []
        
        # 当前时间戳
        self.current_timestamp = 0
    
    def _get_next_order_id(self) -> str:
        """生成订单ID"""
        self._order_counter += 1
        return f"ORD_{self._order_counter}"
    
    def _get_next_trade_id(self) -> str:
        """生成成交ID"""
        self._trade_counter += 1
        return f"TRD_{self._trade_counter}"
    
    def add_order(
        self,
        agent_id: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.LIMIT,
    ) -> Tuple[str, List[Trade]]:
        """
        添加订单并撮合
        
        Returns:
            (order_id, trades)
        """
        order_id = self._get_next_order_id()
        self.current_timestamp += 1
        
        order = Order(
            order_id=order_id,
            agent_id=agent_id,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            timestamp=self.current_timestamp,
        )
        
        self.orders[order_id] = order
        
        # 尝试撮合
        trades = self._match_order(order)
        
        # 如果订单未完全成交，加入订单簿
        if order.quantity > 0 and order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                heapq.heappush(self.bids, (-order.price, order.timestamp, order))
            else:
                heapq.heappush(self.asks, (order.price, order.timestamp, order))
        elif order.quantity > 0:
            # 市价单未完全成交，剩余部分取消
            del self.orders[order_id]
        
        return order_id, trades
    
    def _match_order(self, order: Order) -> List[Trade]:
        """撮合订单"""
        trades = []
        
        if order.side == OrderSide.BUY:
            # 买单：与最低卖价撮合
            while order.quantity > 0 and self.asks:
                best_ask_price, _, best_ask = self.asks[0]
                
                # 检查价格是否匹配
                if order.order_type == OrderType.LIMIT and order.price < best_ask_price:
                    break
                
                # 成交
                heapq.heappop(self.asks)
                trade_quantity = min(order.quantity, best_ask.quantity)
                trade_price = best_ask_price
                
                trade = Trade(
                    trade_id=self._get_next_trade_id(),
                    buyer_id=order.agent_id,
                    seller_id=best_ask.agent_id,
                    price=trade_price,
                    quantity=trade_quantity,
                    timestamp=self.current_timestamp,
                    buy_order_id=order.order_id,
                    sell_order_id=best_ask.order_id,
                )
                trades.append(trade)
                self.trades.append(trade)
                
                # 更新订单数量
                order.quantity -= trade_quantity
                best_ask.quantity -= trade_quantity
                
                # 如果卖单未完全成交，重新放入订单簿
                if best_ask.quantity > 0:
                    heapq.heappush(self.asks, (best_ask.price, best_ask.timestamp, best_ask))
                else:
                    del self.orders[best_ask.order_id]
                    
        else:
            # 卖单：与最高买价撮合
            while order.quantity > 0 and self.bids:
                best_bid_price_neg, _, best_bid = self.bids[0]
                best_bid_price = -best_bid_price_neg
                
                # 检查价格是否匹配
                if order.order_type == OrderType.LIMIT and order.price > best_bid_price:
                    break
                
                # 成交
                heapq.heappop(self.bids)
                trade_quantity = min(order.quantity, best_bid.quantity)
                trade_price = best_bid_price
                
                trade = Trade(
                    trade_id=self._get_next_trade_id(),
                    buyer_id=best_bid.agent_id,
                    seller_id=order.agent_id,
                    price=trade_price,
                    quantity=trade_quantity,
                    timestamp=self.current_timestamp,
                    buy_order_id=best_bid.order_id,
                    sell_order_id=order.order_id,
                )
                trades.append(trade)
                self.trades.append(trade)
                
                # 更新订单数量
                order.quantity -= trade_quantity
                best_bid.quantity -= trade_quantity
                
                # 如果买单未完全成交，重新放入订单簿
                if best_bid.quantity > 0:
                    heapq.heappush(self.bids, (-best_bid.price, best_bid.timestamp, best_bid))
                else:
                    del self.orders[best_bid.order_id]
        
        return trades
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id not in self.orders:
            return False
        
        # 标记为已取消（实际从订单簿中移除在撮合时处理）
        del self.orders[order_id]
        return True
    
    def get_best_bid(self) -> Optional[Tuple[float, float]]:
        """获取最优买价和数量"""
        while self.bids:
            price_neg, _, order = self.bids[0]
            if order.order_id in self.orders:
                return (-price_neg, order.quantity)
            heapq.heappop(self.bids)
        return None
    
    def get_best_ask(self) -> Optional[Tuple[float, float]]:
        """获取最优卖价和数量"""
        while self.asks:
            price, _, order = self.asks[0]
            if order.order_id in self.orders:
                return (price, order.quantity)
            heapq.heappop(self.asks)
        return None
    
    def get_mid_price(self) -> Optional[float]:
        """获取中间价"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return (best_bid[0] + best_ask[0]) / 2
        elif best_bid:
            return best_bid[0]
        elif best_ask:
            return best_ask[0]
        return None
    
    def get_spread(self) -> Optional[float]:
        """获取买卖价差"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask[0] - best_bid[0]
        return None
    
    def get_depth(self) -> Dict[str, List[Tuple[float, float]]]:
        """获取市场深度"""
        bids_depth: Dict[float, float] = {}
        asks_depth: Dict[float, float] = {}
        
        # 统计买单深度
        for price_neg, _, order in self.bids:
            if order.order_id in self.orders:
                price = -price_neg
                bids_depth[price] = bids_depth.get(price, 0) + order.quantity
        
        # 统计卖单深度
        for price, _, order in self.asks:
            if order.order_id in self.orders:
                asks_depth[price] = asks_depth.get(price, 0) + order.quantity
        
        # 排序
        sorted_bids = sorted(bids_depth.items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(asks_depth.items(), key=lambda x: x[0])
        
        return {
            "bids": sorted_bids,
            "asks": sorted_asks,
        }
    
    def get_snapshot(self) -> MarketSnapshot:
        """获取市场快照"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        # 计算24小时成交量
        volume_24h = sum(t.quantity for t in self.trades[-100:]) if self.trades else 0
        
        # 计算总买卖量
        bid_volume = sum(q for _, q in self.get_depth()["bids"])
        ask_volume = sum(q for _, q in self.get_depth()["asks"])
        
        last_price = self.trades[-1].price if self.trades else None
        
        return MarketSnapshot(
            timestamp=self.current_timestamp,
            best_bid=best_bid[0] if best_bid else None,
            best_ask=best_ask[0] if best_ask else None,
            mid_price=self.get_mid_price(),
            spread=self.get_spread(),
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            last_price=last_price,
            volume_24h=volume_24h,
        )


class CarbonMarket:
    """碳金融市场类"""
    
    def __init__(self, config):
        self.config = config
        self.order_book = OrderBook(tick_size=config.market.tick_size)
        
        # 价格限制
        self.price_floor = config.market.price_floor
        self.price_ceiling = config.market.price_ceiling
        
        # 交易费率
        self.transaction_fee_rate = config.market.transaction_fee_rate
        
        # 历史价格
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        
        # 当前步
        self.current_step = 0
    
    def submit_buy_order(
        self,
        agent_id: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.LIMIT,
    ) -> Tuple[str, List[Trade]]:
        """提交买单"""
        # 价格限制检查
        if price is not None:
            price = max(self.price_floor, min(self.price_ceiling, price))
            # 对齐到tick_size
            price = round(price / self.config.market.tick_size) * self.config.market.tick_size
        
        order_id, trades = self.order_book.add_order(
            agent_id=agent_id,
            side=OrderSide.BUY,
            quantity=quantity,
            price=price,
            order_type=order_type,
        )
        
        # 记录成交
        for trade in trades:
            self.price_history.append(trade.price)
            self.volume_history.append(trade.quantity)
        
        return order_id, trades
    
    def submit_sell_order(
        self,
        agent_id: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.LIMIT,
    ) -> Tuple[str, List[Trade]]:
        """提交卖单"""
        # 价格限制检查
        if price is not None:
            price = max(self.price_floor, min(self.price_ceiling, price))
            # 对齐到tick_size
            price = round(price / self.config.market.tick_size) * self.config.market.tick_size
        
        order_id, trades = self.order_book.add_order(
            agent_id=agent_id,
            side=OrderSide.SELL,
            quantity=quantity,
            price=price,
            order_type=order_type,
        )
        
        # 记录成交
        for trade in trades:
            self.price_history.append(trade.price)
            self.volume_history.append(trade.quantity)
        
        return order_id, trades
    
    def get_current_price(self) -> float:
        """获取当前价格
        
        综合最近成交价和订单簿供需信号：
        - 有交易时：以成交价为基础，结合订单簿压力调整
        - 无交易时：用订单簿中间价或初始价格
        """
        last_trade = self.price_history[-1] if self.price_history else None
        mid_price = self.order_book.get_mid_price()
        
        if last_trade is not None:
            # 基于成交价，结合订单簿买卖压力微调
            best_bid = self.order_book.get_best_bid()
            best_ask = self.order_book.get_best_ask()
            if best_bid and best_ask:
                # 买卖压力不对称 → 价格有方向性
                bid_vol = sum(q for _, q in self.order_book.get_depth().get("bids", [])[:5])
                ask_vol = sum(q for _, q in self.order_book.get_depth().get("asks", [])[:5])
                total = bid_vol + ask_vol
                if total > 0:
                    pressure = (bid_vol - ask_vol) / total  # [-1, 1]
                    # 价格微调：买方压力大→价格上移，卖方压力大→价格下移
                    adjusted = last_trade * (1 + pressure * 0.005)
                    return max(self.price_floor, min(self.price_ceiling, adjusted))
            return last_trade
        
        if mid_price is not None:
            return mid_price
        return self.config.market.initial_price
    
    def get_price_change(self, steps: int = 1) -> float:
        """获取价格变化"""
        if len(self.price_history) < steps + 1:
            return 0.0
        return self.price_history[-1] - self.price_history[-(steps + 1)]
    
    def get_volatility(self, window: int = 20) -> float:
        """计算价格波动率"""
        if len(self.price_history) < window:
            return 0.0
        prices = self.price_history[-window:]
        returns = np.diff(prices) / np.array(prices[:-1])
        return np.std(returns) if len(returns) > 0 else 0.0
    
    def step(self):
        """市场步进"""
        self.current_step += 1
        self.order_book.current_timestamp = self.current_step
    
    def get_market_info(self) -> Dict:
        """获取市场信息"""
        snapshot = self.order_book.get_snapshot()
        return {
            "current_price": self.get_current_price(),
            "best_bid": snapshot.best_bid,
            "best_ask": snapshot.best_ask,
            "spread": snapshot.spread,
            "mid_price": snapshot.mid_price,
            "bid_volume": snapshot.bid_volume,
            "ask_volume": snapshot.ask_volume,
            "volume_24h": snapshot.volume_24h,
            "volatility": self.get_volatility(),
            "total_trades": len(self.order_book.trades),
        }
