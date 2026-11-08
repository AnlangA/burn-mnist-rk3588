#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Run as root from a SHA256SUMS-verified digit recognizer payload.
set -euo pipefail
[[ $EUID == 0 ]] || { echo 'run as root' >&2; exit 1; }
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
case ${1:-} in
	--remove)
		systemctl disable --now digit-recognizer.service
		rm -f /etc/systemd/system/digit-recognizer.service /usr/local/bin/digit-recognizer /etc/default/digit-recognizer
		rm -rf /usr/share/digit-recognizer
		systemctl daemon-reload
		systemctl start getty@tty1.service
		echo 'Removed digit recognizer service, binary and samples.'
		exit 0 ;;
	''|--no-start) ;;
	*) echo 'usage: install-digits.sh [--no-start|--remove]' >&2; exit 2 ;;
esac
sha256sum --quiet -c SHA256SUMS
[[ $(uname -m) == aarch64 ]] || { echo 'ARM64 target required' >&2; exit 1; }
for package in libinput10 libudev1 libxkbcommon0 libseat1 libfontconfig1 seatd fonts-dejavu-core fonts-wqy-zenhei libegl-mesa0 libgles2 libgbm1 libgl1-mesa-dri; do
	[[ $(dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null) == installed ]] || {
		echo "missing package: $package; see app/slint-dashboard/README.md" >&2; exit 1;
	}
done
# Reuse the display user provisioned for the dashboard payload.
getent group slint-display >/dev/null || groupadd --system slint-display
id slint-display >/dev/null 2>&1 || useradd --system --gid slint-display \
	--home-dir /nonexistent --no-create-home --shell /usr/sbin/nologin slint-display
backup=$(mktemp -d /var/backups/digit-recognizer.XXXXXXXX)
for file in /usr/local/bin/digit-recognizer /usr/local/libexec/slint-wait-display /etc/systemd/system/digit-recognizer.service /etc/default/digit-recognizer; do
	[[ ! -e $file ]] || cp -a --parents "$file" "$backup"
done
install -m 0755 digit-recognizer /usr/local/bin/digit-recognizer.next
mv -f /usr/local/bin/digit-recognizer.next /usr/local/bin/digit-recognizer
install -m 0644 digit-recognizer.service /etc/systemd/system/digit-recognizer.service
install -D -m 0755 slint-wait-display.sh /usr/local/libexec/slint-wait-display
install -m 0644 digit-recognizer.default /etc/default/digit-recognizer
install -d -m 0755 /usr/share/digit-recognizer/images /usr/share/digit-recognizer/model
rm -f /usr/share/digit-recognizer/images/*.png
install -m 0644 images/*.png /usr/share/digit-recognizer/images/
if [[ -d model ]]; then
	for m in model/*; do
		[[ -f $m ]] && install -m 0644 "$m" "/usr/share/digit-recognizer/model/$(basename "$m")"
	done
fi
if [[ ! -c /dev/rknpu ]]; then
	modprobe rknpu 2>/dev/null || true
fi
if [[ -c /dev/rknpu ]]; then
	chgrp render /dev/rknpu 2>/dev/null || true
	chmod 0666 /dev/rknpu 2>/dev/null || true
fi
systemd-analyze verify /etc/systemd/system/digit-recognizer.service
systemctl daemon-reload
systemctl enable seatd.service digit-recognizer.service
if [[ ${1:-} != --no-start ]]; then
	# Conflicts= stops the dashboard, but wait for it to release the scanout.
	systemctl stop slint-dashboard.service 2>/dev/null || true
	systemctl start seatd.service
	systemctl restart --no-block digit-recognizer.service
fi
echo "Installed digit recognizer; backup: $backup"
