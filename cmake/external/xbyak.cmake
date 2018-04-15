#===============================================================================
# This file is part of the JITInfer (https://github.com/tensor-tangi/jitinfer).
# Copyright (c) 2018 Tensor Tang.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#===============================================================================

include(ExternalProject)

set(XBYAK_PROJECT       extern_xbyak)
set(XBYAK_PREFIX_DIR    ${THIRD_PARTY_PATH}/xbyak)
set(XBYAK_INSTALL_ROOT  ${THIRD_PARTY_INSTALL_PATH}/xbyak)
set(XBYAK_INC_DIR       ${XBYAK_INSTALL_ROOT}/include)

include_directories(${XBYAK_INC_DIR})
include_directories(${XBYAK_INC_DIR}/xbyak)

ExternalProject_Add(
    ${XBYAK_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DEPENDS             ""
    GIT_REPOSITORY      "https://github.com/herumi/xbyak.git"
    GIT_TAG             "v5.64"  # Apr 7th
    PREFIX              ${XBYAK_PREFIX_DIR}
    UPDATE_COMMAND      ""
    CMAKE_ARGS          -DCMAKE_INSTALL_PREFIX=${XBYAK_INSTALL_ROOT}
    CMAKE_CACHE_ARGS    -DCMAKE_INSTALL_PREFIX:PATH=${XBYAK_INSTALL_ROOT}
)

add_library(xbyak INTERFACE)
add_dependencies(xbyak ${XBYAK_PROJECT})
list(APPEND external_project_dependencies xbyak)
