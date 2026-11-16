#!/usr/bin/env bash
# Front-end presets of the WACO runtime. Source this file, then call `waco_preset <name>`.
#
#   basic       the two-anchor front end with a forward-splat image channel and grey fill
#               (LIBERO-PoseShift)
#   wide        the front end for wide viewpoint changes on the nominal-view calibration bundle:
#               dense scoring and plane-constrained global search, inverse warp, unseen pixels
#               from the empty-workspace reference (on LIBERO-PoseShift it behaves like `basic`)
#   liberoplus  `wide` with the in-scene calibration and its twelve-view scan, the floor-alias and
#               plausibility checks and the arm-only fallback (LIBERO-Plus camera track)
waco_preset() {
  unset POSESHIFT_FRONTEND POSESHIFT_WARP POSESHIFT_HOLE_FILL POSESHIFT_CALIB POSESHIFT_MULTIVIEW \
        POSESHIFT_APPEARANCE_GATE POSESHIFT_PLAUSIBLE POSESHIFT_ALIAS POSESHIFT_ARMGLOBAL
  case ${1:-basic} in
    basic) ;;
    wide)
      export POSESHIFT_FRONTEND=wide POSESHIFT_WARP=inverse POSESHIFT_HOLE_FILL=ref ;;
    liberoplus)
      export POSESHIFT_FRONTEND=wide POSESHIFT_WARP=inverse POSESHIFT_HOLE_FILL=ref \
             POSESHIFT_CALIB=calib_liberoplus POSESHIFT_MULTIVIEW=1 POSESHIFT_APPEARANCE_GATE=1 \
             POSESHIFT_PLAUSIBLE=1 POSESHIFT_ALIAS=1 POSESHIFT_ARMGLOBAL=1 ;;
    *) echo "unknown preset: $1 (basic | wide | liberoplus)" >&2; return 2 ;;
  esac
}
