from typing import Dict
import pandas as pd

def _find_col(df: pd.DataFrame, *keywords) -> pd.Series | None:
    """Return first column that contains any of the keywords (case-insensitive)."""
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in keywords):
            return df[col]
    return None



def _check_ddos(df: pd.DataFrame) -> Dict:
    """
    DDoS: extremely high packet/byte rates, very short flows, 
    massive forward packet counts.
    """
    signals = []

    fwd_packets = _find_col(df, "fwd packet", "total fwd", "fwd pkts")
    flow_duration = _find_col(df, "flow duration", "duration")
    pkt_rate = _find_col(df, "packet rate", "pkt rate", "flow pkts/s")
    byte_rate = _find_col(df, "byte rate", "bytes/s", "flow bytes")

    if fwd_packets is not None and fwd_packets.mean() > 0:
        if fwd_packets.max() / max(fwd_packets.mean(), 1e-9) > 50:
            signals.append("extremely high forward packet count")

    if flow_duration is not None and flow_duration.mean() > 0:
        if flow_duration.median() < 100:
            signals.append("very short flow durations typical of flood attacks")

    if pkt_rate is not None:
        if pkt_rate.quantile(0.9) > 10000:
            signals.append("packet rate spike in top 10% of flows")

    if byte_rate is not None:
        if byte_rate.max() / max(byte_rate.mean(), 1e-9) > 30:
            signals.append("extreme byte rate burst detected")

    return {
        "detected": len(signals) >= 1,
        "signals": signals,
        "attack_type": "DDoS"
    }


def _check_dos(df: pd.DataFrame) -> Dict:
    """
    DoS: sustained high traffic from fewer sources,
    large packet lengths, high flow bytes.
    """
    signals = []

    total_len = _find_col(df, "total length", "total fwd len", "total bwd len")
    bwd_packets = _find_col(df, "bwd packet", "total bwd", "bwd pkts")
    flow_bytes = _find_col(df, "flow bytes", "bytes/s")
    pkt_len_mean = _find_col(df, "packet length mean", "pkt len mean", "avg pkt")

    if total_len is not None and total_len.mean() > 0:
        if total_len.mean() > 5000:
            signals.append("high average total packet length suggesting sustained flooding")

    if bwd_packets is not None and bwd_packets.mean() > 0:
        if bwd_packets.max() / max(bwd_packets.mean(), 1e-9) > 40:
            signals.append("large backward packet burst typical of DoS response flooding")

    if flow_bytes is not None:
        if flow_bytes.mean() > 50000:
            signals.append("high mean flow byte count consistent with DoS traffic")

    if pkt_len_mean is not None:
        if pkt_len_mean.mean() > 800:
            signals.append("large mean packet size suggesting bulk data flooding")

    return {
        "detected": len(signals) >= 1,
        "signals": signals,
        "attack_type": "DoS"
    }


def _check_port_scanning(df: pd.DataFrame) -> Dict:
    """
    Port Scanning: many unique destination ports, 
    very small packets, short flows, low byte counts.
    """
    signals = []

    dst_port = _find_col(df, "dst port", "destination port", "dest port")
    pkt_len = _find_col(df, "packet length min", "pkt len min", "min pkt")
    flow_duration = _find_col(df, "flow duration", "duration")
    fwd_packets = _find_col(df, "fwd packet", "total fwd")

    if dst_port is not None:
        unique_ports = dst_port.nunique()
        if unique_ports > 50:
            signals.append(f"high number of unique destination ports ({unique_ports}) suggesting scanning")

    if pkt_len is not None:
        if pkt_len.mean() < 60:
            signals.append("very small minimum packet sizes typical of probe packets")

    if flow_duration is not None:
        if flow_duration.mean() < 50:
            signals.append("very short average flow duration consistent with port probing")

    if fwd_packets is not None:
        if fwd_packets.mean() < 3:
            signals.append("very few packets per flow suggesting single-probe connections")

    return {
        "detected": len(signals) >= 1,
        "signals": signals,
        "attack_type": "Port Scanning"
    }


def _check_brute_force(df: pd.DataFrame) -> Dict:
    """
    Brute Force: repeated connection attempts, 
    consistent small packet sizes, high flow count to same port.
    """
    signals = []

    dst_port = _find_col(df, "dst port", "destination port", "dest port")
    pkt_len_std = _find_col(df, "packet length std", "pkt len std")
    flow_duration = _find_col(df, "flow duration", "duration")
    bwd_len = _find_col(df, "bwd packet length mean", "bwd pkt len mean")

    if dst_port is not None:
        common_ports = [22, 21, 23, 80, 443, 3389, 8080]
        port_counts = dst_port.value_counts()
        for port in common_ports:
            if port in port_counts.index:
                if port_counts[port] / max(len(dst_port), 1) > 0.4:
                    signals.append(
                        f"over 40% of flows target port {port} "
                        f"(common brute force target)"
                    )

    if pkt_len_std is not None:
        if pkt_len_std.mean() < 10:
            signals.append("very low packet length variance suggesting repeated identical requests")

    if flow_duration is not None:
        if 100 < flow_duration.mean() < 5000:
            signals.append("medium-length flows consistent with repeated login attempts")

    if bwd_len is not None:
        if bwd_len.mean() < 100:
            signals.append("small backward packet length suggesting repeated failed responses")

    return {
        "detected": len(signals) >= 1,
        "signals": signals,
        "attack_type": "Brute Force"
    }


def _check_bot(df: pd.DataFrame) -> Dict:
    """
    Bot: periodic traffic patterns, 
    moderate consistent flow rates, mixed port usage.
    """
    signals = []

    iat = _find_col(df, "flow iat mean", "iat mean", "inter arrival")
    iat_std = _find_col(df, "flow iat std", "iat std")
    pkt_rate = _find_col(df, "packet rate", "pkt rate", "flow pkts/s")
    dst_port = _find_col(df, "dst port", "destination port")

    if iat is not None and iat_std is not None:
        if iat.mean() > 0 and (iat_std.mean() / max(iat.mean(), 1e-9)) < 0.3:
            signals.append("very regular inter-arrival times suggesting automated/bot traffic")

    if pkt_rate is not None:
        if 100 < pkt_rate.mean() < 5000:
            signals.append("moderate consistent packet rate typical of bot heartbeat traffic")

    if dst_port is not None:
        unique_ports = dst_port.nunique()
        if 5 < unique_ports < 30:
            signals.append("moderate port diversity consistent with bot C2 communication patterns")

    return {
        "detected": len(signals) >= 1,
        "signals": signals,
        "attack_type": "Bots"
    }


def rule_based_screening(df: pd.DataFrame) -> Dict[str, object]:
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return {
            "rule_flag": False,
            "rule_score": 0.0,
            "reason": "No numeric traffic features found. Only ML detection applied.",
            "suspicious_columns": [],
            "rule_attack_hints": [],
        }

    high_variance_columns = []
    mean_score = 0.0

    for col in numeric_df.columns:
        col_series = numeric_df[col]
        if col_series.mean() == 0:
            continue
        ratio = col_series.max() / max(col_series.mean(), 1e-9)
        if ratio >= 20:
            high_variance_columns.append(col)
        mean_score += ratio

    mean_score = mean_score / max(len(numeric_df.columns), 1)
    rule_flag = len(high_variance_columns) >= 2 or mean_score >= 8

    attack_checks = [
        _check_ddos(numeric_df),
        _check_dos(numeric_df),
        _check_port_scanning(df),   
        _check_brute_force(df),
        _check_bot(df),
    ]

    rule_attack_hints = [
        {
            "attack_type": check["attack_type"],
            "signals": check["signals"]
        }
        for check in attack_checks
        if check["detected"]
    ]

    if rule_attack_hints:
        rule_flag = True

    if rule_attack_hints:
        hint_names = [h["attack_type"] for h in rule_attack_hints]
        reason = (
            f"Rule-based analysis flagged patterns consistent with: "
            f"{', '.join(hint_names)}."
        )
    elif rule_flag:
        reason = (
            "Traffic shows unusually large feature spikes compared to average values."
        )
    else:
        reason = "No strong attack pattern detected by rule-based screening."

    return {
        "rule_flag": rule_flag,
        "rule_score": round(float(mean_score), 2),
        "reason": reason,
        "suspicious_columns": high_variance_columns[:8],
        "rule_attack_hints": rule_attack_hints,  
    }