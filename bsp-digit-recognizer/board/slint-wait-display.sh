#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-only
# Wait without acquiring DRM master or repeatedly failing the application.
set -eu
if [ "${SLINT_DRM_OUTPUT:-}" = list ] || [ "${SLINT_DRM_MODE:-}" = list ]; then
	exit 0
fi
echo 'Waiting for Panthor GPU and a connected DRM output.'
while :; do
	gpu_ready=0
	for driver in /sys/class/drm/renderD*/device/driver; do
		[ -L "$driver" ] || continue
		[ "$(basename "$(readlink -f "$driver")")" = panthor ] || continue
		gpu_ready=1
	done
	if [ "$gpu_ready" = 0 ]; then
		sleep 2
		continue
	fi
	for status in /sys/class/drm/card*-*/status; do
		[ -r "$status" ] || continue
		connector=${status%/status}
		connector=${connector##*/}
		connector=${connector#card*-}
		[ -z "${SLINT_DRM_OUTPUT:-}" ] || [ "$connector" = "$SLINT_DRM_OUTPUT" ] || continue
		if [ "$(cat "$status")" = connected ]; then
			echo "Connected DRM output: $connector"
			exit 0
		fi
	done
	sleep 2
done
