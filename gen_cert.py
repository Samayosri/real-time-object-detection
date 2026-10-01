"""Create a self-signed HTTPS certificate so phones / other PCs can use the camera.

    pip install cryptography
    python gen_cert.py                 # auto-detects this PC's LAN IP
    python gen_cert.py 192.168.1.50    # or add extra IPs/hostnames yourself

Creates cert.pem and key.pem next to app.py. app.py switches to HTTPS on its own.
"""
import datetime
import ipaddress
import socket
import sys

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is actually sent
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


names = {"localhost", "127.0.0.1", lan_ip(), *sys.argv[1:]}
san = []
for n in sorted(names):
    try:
        san.append(x509.IPAddress(ipaddress.ip_address(n)))
    except ValueError:
        san.append(x509.DNSName(n))

key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "local-detector")])
now = datetime.datetime.now(datetime.timezone.utc)
cert = (
    x509.CertificateBuilder()
    .subject_name(subject)
    .issuer_name(issuer)
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(now - datetime.timedelta(days=1))
    .not_valid_after(now + datetime.timedelta(days=365))
    .add_extension(x509.SubjectAlternativeName(san), critical=False)
    .sign(key, hashes.SHA256())
)

with open("key.pem", "wb") as f:
    f.write(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
with open("cert.pem", "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

print("Created cert.pem and key.pem for:", ", ".join(sorted(names)))
print(f"Now run python app.py and open https://{lan_ip()}:8000 on your phone.")
