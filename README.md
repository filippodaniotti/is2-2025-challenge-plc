# Conditioning PARCNet with TFiLM for Robust Packet Loss Concealment

Real-time applications, such as Networked Music Performance (NMP), typically employ best-effort protocols to minimize latency; however, this may cause buffer underruns at the receiver side. Packet Loss Concealment (PLC) techniques are employed to cope with this issue. While a few PLC algorithms have been proposed over the years, they usually assume that the past buffer always features valid packets, which is usually not the case in real life scenarios. In this technical report, we present a novel method to address the increasing divergence of the next prediction when the past buffer features packets that are prediction themselves. We iteratively train several instances of our model, and for each iteration an increasing number of packets in the buffer is dropped and concealed with a surrogate model, ensuring robust concealments even when the past buffer is lossy. Additionally, we employ use Temporal Feature-Wise Linear Modulation (TFiLM) as a conditioning strategy to leverage the positional information of concealed packets in the buffer.

## Resources

Processed audio clips: https://www.dropbox.com/scl/fi/8wfrn5vop0qyj3mjwrzvm/daniotti-is2_2025_enhanced.zip?rlkey=icloc24kq6cviab8cnmo4onx4&st=tfkdq779&dl=0
