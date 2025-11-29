DISABLE_MISSING_PROFILE_WARNING()
INCLUDE_DIRECTORIES(SYSTEM)

SET(GEMBED_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/gembed")
SET(GEMBED_LIB_PATH "${GEMBED_ROOT}/target/release/libgembed.a")

ADD_CUSTOM_TARGET(build_rust_gembed
  COMMAND cargo build --release
  WORKING_DIRECTORY ${GEMBED_ROOT}
  COMMENT "Compiling Rust Gembed library..."
)

# Platform Specifics (macOS)
SET(EXTRA_LIBS pthread ${CMAKE_DL_LIBS})

IF(APPLE)
  LIST(APPEND EXTRA_LIBS
    "-framework Foundation"
    "-framework Security"
    "-framework CoreFoundation"
    "-framework CoreML"
    "-framework Accelerate"
    "-lobjc"
  )
ENDIF()

MYSQL_ADD_COMPONENT(mysql_gembed
  mysql_gembed.cc
  MODULE_ONLY
  TEST_ONLY
  LINK_LIBRARIES ${GEMBED_LIB_PATH} ${EXTRA_LIBS}
)

ADD_DEPENDENCIES(component_mysql_gembed build_rust_gembed)
